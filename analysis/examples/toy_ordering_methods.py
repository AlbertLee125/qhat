"""Toy demonstration of Pauli ordering methods.

Run from the repository root:

    python analysis/examples/toy_ordering_methods.py
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = REPO_ROOT.parent
for path in (REPO_PARENT, REPO_ROOT):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from qhat.analysis.ordering import reorder_paulis


TOY_HAMILTONIAN = {
    "ZZ": 0.2,
    "XI": -1.5,
    "IY": 0.7,
    "IZ": -0.1,
    "XX": 2.0,
    "II": 0.05,
}


METHODS = [
    "identity",
    "reverse",
    "magnitude_descending",
    "magnitude_ascending",
    "lexicographical",
    "pauli_weight_ascending",
    "pauli_weight_descending",
    "diagonal_first",
    "diagonal_last",
    "commuting_blocks",
    "random_seeded",
]


def format_terms(ordered_terms):
    return " -> ".join(
        f"{pauli_string}({coefficient:+.2f})"
        for pauli_string, coefficient in ordered_terms.items()
    )


def main():
    print("Toy Hamiltonian:")
    print(format_terms(TOY_HAMILTONIAN))
    print()

    for method in METHODS:
        seed = 7 if method == "random_seeded" else None
        ordered_terms = reorder_paulis(TOY_HAMILTONIAN, method, seed=seed)
        print(f"{method:24s} {format_terms(ordered_terms)}")


if __name__ == "__main__":
    main()
