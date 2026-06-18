# Ordering Methods Roadmap

This note is a follow-up proposal for `analysis/ordering.py` after the
`upstream/improve_ordering` cleanup branch lands. That branch fixes important
edge cases in the current ordering function, including empty inputs, default
ordering return type, identity-only Pauli terms, and the `Y` branch typo in
`group_evolve_xyz`.

The next step should not be another long `if`/`elif` patch. The scientific
problem is open-ended: for molecular Hamiltonians and spin Hamiltonians, there
is no universally superior Pauli-term ordering. A useful implementation should
make it easy to add and compare ordering strategies without changing the public
Trotterization code each time.

## Why More Orderings Are Needed

Jordan-Wigner and Bravyi-Kitaev are isomorphic fermion-to-qubit encodings, so
the exact Hamiltonian eigenvalues should agree up to numerical precision when
they are built from the same molecular problem. Their Trotterized evolutions can
still behave differently because the encodings produce different Pauli strings,
Pauli weights, coefficient distributions, and commutation graphs.

For a first-order product formula, the leading Baker-Campbell-Hausdorff error
contains commutators between Hamiltonian terms. A simple norm bound may ignore
ordering, but the actual error operator can change because reordering changes
the signs and cancellation pattern of those commutators. For higher-order
formulas, nested commutators make the ordering dependence even more visible.

This means ordering methods are not just software conveniences. They are
experimental controls for understanding why JW and BK can have different
Trotter errors even when their exact spectra agree.

## Implementation Direction

The stable API should be a registry of named ordering strategies:

```python
ORDERING_METHODS = {
    "identity": identity,
    "magnitude_descending": magnitude_descending,
    "magnitude_ascending": magnitude_ascending,
    "lexicographical": lexicographical,
    "pauli_weight_ascending": pauli_weight_ascending,
    "pauli_weight_descending": pauli_weight_descending,
    "diagonal_first": diagonal_first,
    "commuting_blocks": commuting_blocks,
    "random_seeded": random_seeded,
    "group_evolve_xyz": group_evolve_xyz,
}
```

Each strategy should accept materialized Pauli terms and return the same terms
in a new order. The ordering function should validate input once, look up the
strategy by name, and call it. New methods then become local strategy functions
or registered plugins rather than new branches in a growing conditional.

Recommended contract:

```python
def ordering_method(terms, *, seed=None):
    """Return the same Pauli terms in a deterministic or seeded order."""
```

All methods should preserve every input term exactly once. They should not
change coefficients, combine terms, drop identities, or depend on dictionary
insertion order unless the method is explicitly named `identity`.

## First-Wave Methods

| Method | Add now? | Why it is scientifically useful |
| --- | --- | --- |
| `identity` | Yes | Gives an explicit no-reorder baseline. This is better than using `None` in experiments because plots and logs can name the baseline consistently. |
| `reverse` | Yes | Provides a paired control for any insertion order. If `identity` and `reverse` differ strongly, the Trotter result is highly order-sensitive. |
| `magnitude_descending` | Yes | Keeps the existing `magnitude` idea but gives it a precise name. Large coefficients often dominate commutator terms, so this is a natural first heuristic. |
| `magnitude_ascending` | Yes | The mirror case is needed because descending magnitude is plausible, not guaranteed. It helps distinguish real improvement from an arbitrary convention. |
| `lexicographical` | Yes | Provides a deterministic canonical ordering independent of coefficient size. This is valuable for reproducibility and regression tests. |
| `pauli_weight_ascending` | Yes | Sorts by the number of non-identity Pauli operators. JW and BK often differ in Pauli weight, so this directly probes encoding-dependent locality. |
| `pauli_weight_descending` | Yes | The mirror case tests whether long strings should be handled early or late. It also helps expose sensitivity to high-weight BK or JW terms. |
| `diagonal_first` | Yes | Moves `I`/`Z`-only terms before non-diagonal terms. Many chemistry Hamiltonians contain diagonal number-operator structure, and diagonal terms often commute with each other. |
| `diagonal_last` | Yes | The paired control for `diagonal_first`. If the two differ, diagonal/non-diagonal placement is affecting commutator cancellation. |
| `commuting_blocks` | Yes, but after tests are added | Groups mutually commuting terms into stable blocks. Terms inside a commuting block can be ordered without creating internal Trotter error. |
| `anticommutation_degree` | Yes, but after tests are added | Sorts terms by how many other terms they anticommute with, optionally weighted by coefficient magnitude. This directly targets the commutation graph behind Trotter error. |
| `random_seeded` | Yes, but only with a seed | Gives a statistical baseline. It prevents overclaiming that one deterministic heuristic is superior when the Hamiltonian may simply be order-sensitive. |
| `group_evolve_xyz` | Keep as specialized | Useful for Hamiltonians whose terms contain at most one non-identity Pauli type. It should stay, but its precondition is too narrow to serve as a general chemistry ordering. |

## Practical Priority

The most solid initial expansion is:

1. `identity`
2. `reverse`
3. `magnitude_descending`
4. `magnitude_ascending`
5. `lexicographical`
6. `pauli_weight_ascending`
7. `pauli_weight_descending`
8. `diagonal_first`
9. `diagonal_last`
10. `random_seeded`

These are simple, deterministic or seeded, easy to test, and do not require a
heavy graph algorithm. They are enough to answer early questions such as:

- Do JW and BK have the same exact eigenvalues for the same molecular system?
- Does Trotter error change under different term orderings?
- Is the difference mostly explained by coefficient size, Pauli weight,
  diagonal structure, or general order sensitivity?

`commuting_blocks` and `anticommutation_degree` should follow once the test
suite has utilities for computing Pauli commutation relations. They are more
scientifically targeted, but they also introduce more implementation choices.

## Suggested Test Cases

Each method should have small unit tests before it is used in molecular runs:

- empty input returns empty output
- identity-only input is preserved
- all input keys and coefficients are preserved exactly
- deterministic methods return the same order on repeated calls
- seeded random ordering is reproducible with the same seed
- invalid methods report the list of available methods
- `group_evolve_xyz` rejects mixed Pauli-type strings with a clear error

For scientific validation, compare JW and BK on small systems first:

- `H2` in `sto-6g`
- `Be-H` in `sto-6g`
- `Li-Li` with the available Hamiltonian archive inputs

For each system and encoding, record:

- exact eigenvalue agreement between JW and BK
- Trotterized eigenvalue or phase error for each ordering
- number of terms
- Pauli weight distribution
- number of anticommuting term pairs
- norm-weighted commutator score

This is the minimum metadata needed to explain why one ordering behaves better
or worse. Without these measurements, the comparison risks becoming a list of
numerical outcomes without a mechanism.

## Expected Outcome

The goal is not to declare one ordering universally best. The goal is to create
a controlled ordering interface that lets QHA/Trotter experiments separate four
effects:

1. exact encoding equivalence
2. coefficient-size ordering
3. Pauli-locality ordering
4. commutation-graph ordering

That structure gives future JW-versus-BK comparisons a clean path from exact
spectrum checks to Trotter error diagnosis.
