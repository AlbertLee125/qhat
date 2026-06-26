"""
Graph-coloring demo for Pauli-string ordering.

This example is intentionally small and teacherly. It uses the H2/STO-3G
Jordan-Wigner Pauli strings from local_configs/h2_ordering_report.txt and the
helpers in qhat.analysis.ordering to show how graph coloring reduces the
ordering problem.

Run from the repository root with:

    python analysis/examples/group_coloring_ordering_demo.py

The core idea:

    node = one Pauli string term
    edge = two Pauli strings do NOT commute
    color = one pairwise-commuting group

So graph coloring gives groups of Pauli strings that can be moved around inside
the group without changing the product formula.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb, factorial
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import networkx as nx

# Make the file runnable directly from the repository root or from this
# directory, while still preferring normal qhat imports when the package is
# installed.
_REPO_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "analysis" / "ordering.py").exists()
)
for _path in (_REPO_ROOT.parent, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from qhat.analysis.ordering import (
        chromatic_number_greedy,
        create_commutativity_graph,
        is_pauli_commuting,
        reorder_paulis,
    )
except ModuleNotFoundError:
    # Convenience fallback for running this file directly from the repository root.
    from analysis.ordering import (
        chromatic_number_greedy,
        create_commutativity_graph,
        is_pauli_commuting,
        reorder_paulis,
    )


# Dense 4-qubit JW strings copied from local_configs/h2_ordering_report.txt.
# The identity is written as "IIII" here because ordering.py expects dense
# strings of equal length.
H2_STO3G_JW_TERMS: dict[str, float] = {
    "IIII": -9.8863969335e-02,
    "XXYY": -4.5322202053e-02,
    "XYYX": +4.5322202053e-02,
    "YXXY": +4.5322202053e-02,
    "YYXX": -4.5322202053e-02,
    "ZIII": +1.7119774903e-01,
    "ZZII": +1.6862219159e-01,
    "ZIZI": +1.2054482205e-01,
    "ZIIZ": +1.6586702411e-01,
    "IZII": +1.7119774903e-01,
    "IZZI": +1.6586702411e-01,
    "IZIZ": +1.2054482205e-01,
    "IIZI": -2.2278593040e-01,
    "IIZZ": +1.7434844186e-01,
    "IIIZ": -2.2278593040e-01,
}


PauliTerm = tuple[str, float]
PauliGroup = list[PauliTerm]


def h2_sto3g_jw_terms(include_identity: bool = False) -> dict[str, float]:
    """Return the toy H2/STO-3G JW terms used in this demo."""
    terms = dict(H2_STO3G_JW_TERMS)
    if not include_identity:
        terms.pop("IIII")
    return terms


def format_dense_pauli(pauli_string: str) -> str:
    """Format a dense Pauli string as a sparse human-readable label."""
    ops = [
        f"{pauli}{index}"
        for index, pauli in enumerate(pauli_string)
        if pauli != "I"
    ]
    return "I" if not ops else " ".join(ops)


def noncommuting_pairs(terms: Mapping[str, float]) -> list[tuple[PauliTerm, PauliTerm]]:
    """Return all unordered pairs of terms that do not commute."""
    items = list(terms.items())
    pairs = []
    for i, left in enumerate(items):
        for right in items[i + 1:]:
            if not is_pauli_commuting(left[0], right[0], pauli_string_format="explicit"):
                pairs.append((left, right))
    return pairs


def build_greedy_coloring_groups(
    terms: Mapping[str, float],
    strategy: str = "largest_first",
) -> tuple[nx.Graph, dict[int, int], list[PauliGroup]]:
    """
    Build the noncommutation graph and group terms by greedy graph coloring.

    create_commutativity_graph is the existing QHAT helper name, but notice the
    implementation adds an edge when two Pauli strings do NOT commute. That is
    exactly the graph we want to color: adjacent nodes cannot share a color, so
    each color class is a commuting group.
    """
    items = list(terms.items())
    graph = create_commutativity_graph(items, pauli_string_format="explicit")
    coloring = nx.greedy_color(graph, strategy=strategy)

    groups_by_color: dict[int, PauliGroup] = defaultdict(list)
    for index, color in sorted(coloring.items(), key=lambda pair: (pair[1], pair[0])):
        groups_by_color[color].append(items[index])

    groups = [groups_by_color[color] for color in sorted(groups_by_color)]
    return graph, coloring, groups


def group_is_pairwise_commuting(group: Sequence[PauliTerm]) -> bool:
    """Return True if every pair inside a group commutes."""
    for i, left in enumerate(group):
        for right in group[i + 1:]:
            if not is_pauli_commuting(left[0], right[0], pauli_string_format="explicit"):
                return False
    return True


def flatten_groups(groups: Iterable[Sequence[PauliTerm]]) -> list[PauliTerm]:
    """Flatten color groups into the order used by a grouped Trotter product."""
    return [term for group in groups for term in group]


def try_build_commuting_evolution_blocks(
    groups: Sequence[Sequence[PauliTerm]],
    time: float = 1.0,
):
    """
    Build one CommutingPauliStringEvolution block per color group if available.

    This is optional in the demo because the lightweight graph-coloring lesson
    only needs ordering.py and NetworkX. The circuit object currently needs
    heavyweight dependencies such as Qualtran.
    """
    try:
        from qhat.common.commuting_pauli_string_evolution import (
            CommutingPauliStringEvolution,
        )
    except ModuleNotFoundError:
        try:
            from common.commuting_pauli_string_evolution import (
                CommutingPauliStringEvolution,
            )
        except ModuleNotFoundError as exc:
            return None, exc

    blocks = [
        CommutingPauliStringEvolution(pauli_terms=tuple(group), time=time)
        for group in groups
    ]
    return blocks, None


def print_terms(title: str, terms: Sequence[PauliTerm]) -> None:
    """Pretty-print an ordered list of Pauli terms."""
    print(title)
    print("-" * len(title))
    for index, (pauli_string, coefficient) in enumerate(terms):
        print(
            f"{index:2d}: {pauli_string:4s}  "
            f"{format_dense_pauli(pauli_string):15s}  "
            f"{coefficient:+.10e}"
        )
    print()


def main() -> None:
    terms = h2_sto3g_jw_terms(include_identity=False)
    items = list(terms.items())
    total_pairs = len(items) * (len(items) - 1) // 2
    bad_pairs = noncommuting_pairs(terms)

    print("H2/STO-3G Pauli Ordering By Graph Coloring")
    print("==========================================")
    print("Identity is omitted because it commutes with everything and only adds a global phase.")
    print()
    print_terms("Toy Hamiltonian Terms", items)

    print("Pairwise Commutation Summary")
    print("----------------------------")
    print(f"non-identity Pauli terms : {len(items)}")
    print(f"all unordered pairs      : {total_pairs}")
    print(f"commuting pairs          : {total_pairs - len(bad_pairs)}")
    print(f"noncommuting pairs/edges : {len(bad_pairs)}")
    print()

    print("Noncommuting Edges")
    print("------------------")
    for left, right in bad_pairs:
        print(f"{format_dense_pauli(left[0]):15s}  <->  {format_dense_pauli(right[0])}")
    print()

    graph, coloring, groups = build_greedy_coloring_groups(terms)
    greedy_colors = chromatic_number_greedy(graph)

    print("Greedy Graph Coloring")
    print("---------------------")
    print("QHAT helper used          : create_commutativity_graph(...)")
    print("Important interpretation  : graph edges mean NONcommutation")
    print(f"graph nodes               : {graph.number_of_nodes()}")
    print(f"graph edges               : {graph.number_of_edges()}")
    print(f"greedy color count        : {greedy_colors}")
    print()

    for color, group in enumerate(groups):
        status = "yes" if group_is_pairwise_commuting(group) else "NO"
        print(f"Color {color}: {len(group)} terms, pairwise commuting = {status}")
        for pauli_string, coefficient in group:
            print(
                f"  {pauli_string:4s}  "
                f"{format_dense_pauli(pauli_string):15s}  "
                f"{coefficient:+.10e}"
            )
        print()

    grouped_order = flatten_groups(groups)
    print_terms("Order Produced By The Color Groups", grouped_order)

    qhat_reordered = list(reorder_paulis(terms, "group_evolve_greedy").items())
    print_terms('Order Produced By reorder_paulis(..., "group_evolve_greedy")', qhat_reordered)

    print("Search-Space Intuition")
    print("----------------------")
    print(f"naive non-identity permutations : 14! = {factorial(14):,}")
    if len(groups) == 2:
        group_a, group_b = groups
        interleavings = comb(len(group_a) + len(group_b), len(group_b))
        print(
            "if terms inside each color are treated as freely swappable, "
            f"two-color interleavings = C(14, {len(group_b)}) = {interleavings:,}"
        )
    print(f"if each color is kept as one contiguous block, block orders = {factorial(len(groups))}")
    print()

    blocks, error = try_build_commuting_evolution_blocks(groups)
    print("Optional CommutingPauliStringEvolution Blocks")
    print("---------------------------------------------")
    if error is not None:
        print("Skipped block construction because an optional circuit dependency is missing.")
        print(f"Import error: {error}")
    else:
        print(f"Built {len(blocks)} commuting evolution blocks.")
        for index, block in enumerate(blocks):
            print(f"  block {index}: {block.num_terms} commuting terms")


if __name__ == "__main__":
    main()
