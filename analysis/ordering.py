"""Pauli-term ordering strategies for Trotterized time evolution."""

from random import Random


def reorder_paulis(pauli_strings, ordering_method, seed=None):
    """
    Reorder Pauli-string terms for Trotterized time evolution.

    Parameters
    ----------
    pauli_strings:
        Mapping from dense Pauli strings to coefficients, for example
        {"XII": 0.5, "ZZI": -1.2}.
    ordering_method:
        Name of the ordering strategy. If None, the input order is preserved.
    seed:
        Optional integer seed used by seeded stochastic ordering methods.

    Returns
    -------
    dict
        Dictionary containing the same Pauli terms and coefficients, inserted
        in the selected order.
    """
    terms = list(pauli_strings.items())
    _validate_terms(terms)

    if ordering_method is None:
        ordering_method = "identity"

    try:
        ordering_function = ORDERING_METHODS[ordering_method]
    except KeyError as exc:
        available = ", ".join(available_ordering_methods())
        raise ValueError(
            f"The Trotter ordering method {ordering_method!r} is not currently "
            f"supported. Available methods: {available}"
        ) from exc

    return dict(ordering_function(terms, seed=seed))


def available_ordering_methods():
    """Return supported Pauli ordering method names."""
    return tuple(ORDERING_METHODS)


def identity(terms, seed=None):
    """Preserve the input term order."""
    return list(terms)


def reverse(terms, seed=None):
    """Reverse the input term order."""
    return list(reversed(terms))


def magnitude(terms, seed=None):
    """Backward-compatible alias for descending coefficient magnitude."""
    return magnitude_descending(terms, seed=seed)


def magnitude_descending(terms, seed=None):
    """Sort terms by descending coefficient magnitude."""
    return sorted(terms, key=lambda term: abs(term[1]), reverse=True)


def magnitude_ascending(terms, seed=None):
    """Sort terms by ascending coefficient magnitude."""
    return sorted(terms, key=lambda term: abs(term[1]))


def lexicographical(terms, seed=None):
    """
    Sort lexicographically by dense Pauli string.

    Python's string ordering gives I < X < Y < Z for the dense Pauli alphabet.
    """
    return sorted(terms, key=lambda term: term[0])


def pauli_weight_ascending(terms, seed=None):
    """Sort terms from most local to least local."""
    return sorted(terms, key=lambda term: (_pauli_weight(term[0]), term[0]))


def pauli_weight_descending(terms, seed=None):
    """Sort terms from least local to most local."""
    return sorted(terms, key=lambda term: (-_pauli_weight(term[0]), term[0]))


def diagonal_first(terms, seed=None):
    """Move I/Z-only terms before terms containing X or Y."""
    return sorted(terms, key=lambda term: (not _is_diagonal(term[0]), term[0]))


def diagonal_last(terms, seed=None):
    """Move terms containing X or Y before I/Z-only terms."""
    return sorted(terms, key=lambda term: (_is_diagonal(term[0]), term[0]))


def random_seeded(terms, seed=None):
    """Return a reproducible random ordering of terms."""
    if seed is None:
        raise ValueError("random_seeded ordering requires an explicit seed")

    ordered_terms = list(terms)
    Random(seed).shuffle(ordered_terms)
    return ordered_terms


def commuting_blocks(terms, seed=None):
    """
    Greedily group mutually commuting Pauli strings into contiguous blocks.

    The block construction is deterministic: terms are visited in input order,
    and each term is placed into the first existing block with which it commutes
    term-by-term. If no such block exists, a new block is started.
    """
    blocks = []
    for term in terms:
        pauli_string = term[0]
        for block in blocks:
            if all(pauli_commute(pauli_string, block_term[0]) for block_term in block):
                block.append(term)
                break
        else:
            blocks.append([term])

    return [term for block in blocks for term in block]


def group_evolve_xyz(terms, seed=None):
    """
    Group terms that contain only X, only Y, only Z, or only identity operators.

    This method is specialized for Hamiltonians where each Pauli string has at
    most one non-identity Pauli type.
    """
    xs = []
    ys = []
    zs = []

    for term in terms:
        pauli_string = term[0]
        pauli_types = set(pauli_string)
        pauli_types.discard("I")

        if len(pauli_types) > 1:
            raise ValueError(
                "Cannot use this method, group_evolve_xyz can only be used if "
                "every pauli term has at most one non-identity pauli type, but "
                f"this Hamiltonian has the string {pauli_string}"
            )

        if len(pauli_types) == 0:
            xs.append(term)
            continue

        pauli_type = list(pauli_types)[0]
        if pauli_type == "X":
            xs.append(term)
        elif pauli_type == "Y":
            ys.append(term)
        elif pauli_type == "Z":
            zs.append(term)
        else:
            raise ValueError(
                f"Unsupported Pauli type: {pauli_type}. The only allowable "
                "Pauli types are I, X, Y, Z."
            )

    return xs + ys + zs


def pauli_commute(first, second):
    """Return True if two dense Pauli strings commute."""
    if len(first) != len(second):
        raise ValueError(
            f"Pauli strings must have the same length, got {first!r} and {second!r}"
        )

    anticommutes = 0
    for left, right in zip(first, second):
        if left != "I" and right != "I" and left != right:
            anticommutes += 1

    return anticommutes % 2 == 0


def _validate_terms(terms):
    valid_paulis = {"I", "X", "Y", "Z"}
    for pauli_string, _ in terms:
        if not isinstance(pauli_string, str):
            raise ValueError(
                "This method currently only accepts Pauli strings written in "
                "'string' format, e.g. XIIYZIX"
            )

        invalid_symbols = set(pauli_string) - valid_paulis
        if invalid_symbols:
            symbols = ", ".join(sorted(invalid_symbols))
            raise ValueError(
                f"Unsupported Pauli symbol(s) in {pauli_string!r}: {symbols}. "
                "The only allowable Pauli symbols are I, X, Y, Z."
            )


def _pauli_weight(pauli_string):
    return sum(pauli != "I" for pauli in pauli_string)


def _is_diagonal(pauli_string):
    return all(pauli in {"I", "Z"} for pauli in pauli_string)


ORDERING_METHODS = {
    "identity": identity,
    "reverse": reverse,
    "magnitude": magnitude,
    "magnitude_descending": magnitude_descending,
    "magnitude_ascending": magnitude_ascending,
    "lexicographical": lexicographical,
    "pauli_weight_ascending": pauli_weight_ascending,
    "pauli_weight_descending": pauli_weight_descending,
    "diagonal_first": diagonal_first,
    "diagonal_last": diagonal_last,
    "random_seeded": random_seeded,
    "commuting_blocks": commuting_blocks,
    "group_evolve_xyz": group_evolve_xyz,
}
