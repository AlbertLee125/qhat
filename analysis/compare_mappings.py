"""
Compare Jordan-Wigner and Bravyi-Kitaev mappings.

This module focuses on two questions:

1. Do JW and BK produce isospectral qubit Hamiltonians for the same fermionic
   Hamiltonian?
2. How do their Pauli-string structures and Trotter error coefficients differ?

For a complete JW/BK comparison, start from a second-quantized Hamiltonian
(`.npz`, `.npy`, `.h5`, or `.hdf5`) so both mappings can be generated from the
same physical operator.  If you already have mapped Pauli files, provide both
the JW and BK files; a single Pauli file cannot be inverted to recover the
other mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openfermion import (
    FermionOperator,
    InteractionOperator,
    QubitOperator,
    bravyi_kitaev,
    count_qubits,
    jordan_wigner,
)
from openfermion.linalg import get_sparse_operator

from qhat.analysis.config_types import HamiltonianConfiguration
from qhat.analysis.hamiltonian import (
    Hamiltonian,
    LinearCombinationOfPauliStrings,
    dense_to_sparse_pauli,
    get_physical_hamiltonian,
)
from qhat.analysis.mapping_isomorphism import (
    FermionMappingSpectrumComparison,
    compare_jordan_wigner_bravyi_kitaev_spectra,
)
from qhat.analysis.ordering import reorder_paulis
from qhat.analysis.trotter_coefficients_fast import trotter_error_estimator_fast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PauliStructureSummary:
    """Simple structural summary of a mapped Pauli Hamiltonian."""

    mapping: str
    num_qubits: int
    num_terms: int
    num_non_identity_terms: int
    identity_coefficient: float
    one_norm: float
    non_identity_one_norm: float
    coefficient_two_norm: float
    max_pauli_weight: int
    mean_pauli_weight: float
    weight_histogram: dict[int, int]
    x_count: int
    y_count: int
    z_count: int
    commuting_pairs: int
    commuting_density: float
    qubit_wise_commuting_pairs: int
    qubit_wise_commuting_density: float
    anticommute_pairs: int
    total_pairs: int
    anticommute_density: float


@dataclass(frozen=True)
class TrotterCoefficientSummary:
    """Trotter coefficients and QHAT-style step estimates for one mapping."""

    mapping: str
    ordering_method: str | None
    random_seed: int | None
    c1: float
    c2: float
    time: float
    energy_error: float | None
    first_order_steps_estimate: int | None
    second_order_steps_estimate: int | None
    first_order_bound_at_steps: dict[int, float]
    second_order_bound_at_steps: dict[int, float]


@dataclass(frozen=True)
class TrotterActualErrorSummary:
    """Dense exact-vs-Trotter operator errors for small mapped Hamiltonians."""

    mapping: str
    ordering_method: str | None
    random_seed: int | None
    state_label: str | None
    time: float
    first_order_operator_error: dict[int, float]
    second_order_operator_error: dict[int, float]
    first_order_frobenius_error: dict[int, float]
    second_order_frobenius_error: dict[int, float]
    first_order_state_error: dict[int, float] | None
    second_order_state_error: dict[int, float] | None
    first_order_state_infidelity: dict[int, float] | None
    second_order_state_infidelity: dict[int, float] | None


@dataclass(frozen=True)
class MappingReport:
    """Complete comparison result."""

    label: str
    jw_structure: PauliStructureSummary
    bk_structure: PauliStructureSummary | None
    jw_trotter: TrotterCoefficientSummary
    bk_trotter: TrotterCoefficientSummary | None
    jw_actual_errors: TrotterActualErrorSummary | None
    bk_actual_errors: TrotterActualErrorSummary | None
    spectrum: FermionMappingSpectrumComparison | None
    pauli_spectrum_max_abs_difference: float | None
    notes: tuple[str, ...]

    @property
    def is_isospectral(self) -> bool | None:
        if self.spectrum is not None:
            return self.spectrum.is_isospectral
        if self.pauli_spectrum_max_abs_difference is not None:
            return bool(self.pauli_spectrum_max_abs_difference <= 1.0e-10)
        return None

    def as_dict(self):
        data = asdict(self)
        if self.spectrum is not None:
            data["spectrum"] = self.spectrum.as_dict()
        data["is_isospectral"] = self.is_isospectral
        return data


def _load_second_quantized(filename: str, mapping: str) -> Hamiltonian:
    config = HamiltonianConfiguration()
    config.load_second_quantization(
        filename,
        fermion_to_qubit_transform=mapping,
    )
    return get_physical_hamiltonian(config)


def _load_pauli(filename: str) -> Hamiltonian:
    config = HamiltonianConfiguration()
    config.load_pauli_strings(filename)
    return get_physical_hamiltonian(config)


def _infer_fermionic_num_qubits(
    fermionic_hamiltonian: FermionOperator | InteractionOperator,
    num_qubits: int | None,
) -> int:
    if num_qubits is not None:
        if num_qubits < 0:
            raise ValueError("num_qubits must be non-negative.")
        return int(num_qubits)
    if hasattr(fermionic_hamiltonian, "n_qubits"):
        return int(fermionic_hamiltonian.n_qubits)
    return int(count_qubits(fermionic_hamiltonian))


def _mapped_qubit_hamiltonian(
    fermionic_hamiltonian: FermionOperator | InteractionOperator,
    mapping: str,
    *,
    num_qubits: int | None = None,
) -> Hamiltonian:
    resolved_num_qubits = _infer_fermionic_num_qubits(fermionic_hamiltonian, num_qubits)
    mapping_upper = mapping.upper()
    if mapping_upper == "JW":
        qubit_operator = jordan_wigner(fermionic_hamiltonian)
    elif mapping_upper == "BK":
        qubit_operator = bravyi_kitaev(
            fermionic_hamiltonian,
            n_qubits=resolved_num_qubits,
        )
    else:
        raise ValueError(f"Unsupported mapping {mapping!r}; expected 'JW' or 'BK'.")
    qubit_operator.compress()
    return Hamiltonian(
        LinearCombinationOfPauliStrings(
            num_qubits=resolved_num_qubits,
            sparse=dict(qubit_operator.terms),
        )
    )


def _normalize_ordering_method(ordering_method: str | None) -> str | None:
    if ordering_method in (None, "", "native", "input", "input order", "none"):
        return None
    return ordering_method


def _ordering_label(ordering_method: str | None) -> str:
    return _normalize_ordering_method(ordering_method) or "input order"


def _pauli_weight(pauli_key) -> int:
    return len(pauli_key)


def _terms_to_qubit_operator(pauli_terms: dict) -> QubitOperator:
    operator = QubitOperator()
    for pauli_key, coefficient in pauli_terms.items():
        operator += QubitOperator(pauli_key, coefficient)
    operator.compress()
    return operator


def _ordered_dense_items(
    hamiltonian: Hamiltonian,
    ordering_method: str | None = None,
    random_seed: int | None = None,
) -> list[tuple[str, float]]:
    ordering_method = _normalize_ordering_method(ordering_method)
    dense_terms = hamiltonian.get_all_pauli_strings(return_as="strings")
    if ordering_method == "random":
        ordered = list(dense_terms.items())
        rng = np.random.default_rng(random_seed)
        rng.shuffle(ordered)
        return ordered
    ordered = reorder_paulis(dense_terms, ordering_method)
    if isinstance(ordered, dict):
        return list(ordered.items())
    return list(ordered)


def _ordered_sparse_items(
    hamiltonian: Hamiltonian,
    ordering_method: str | None = None,
    random_seed: int | None = None,
) -> list[tuple[tuple, float]]:
    return [
        (dense_to_sparse_pauli(pauli_string), coefficient)
        for pauli_string, coefficient in _ordered_dense_items(
            hamiltonian,
            ordering_method,
            random_seed,
        )
    ]


def _grouped_terms(
    hamiltonian: Hamiltonian,
    ordering_method: str | None = None,
    random_seed: int | None = None,
) -> list[QubitOperator]:
    return [
        QubitOperator(pauli_key, coefficient)
        for pauli_key, coefficient in _ordered_sparse_items(
            hamiltonian,
            ordering_method,
            random_seed,
        )
    ]


def _pauli_anticommute(pauli_a, pauli_b) -> bool:
    ops_a = dict(pauli_a)
    ops_b = dict(pauli_b)
    differences = 0
    for qubit in set(ops_a).intersection(ops_b):
        if ops_a[qubit] != ops_b[qubit]:
            differences += 1
    return differences % 2 == 1


def _pauli_qubit_wise_commute(pauli_a, pauli_b) -> bool:
    ops_a = dict(pauli_a)
    ops_b = dict(pauli_b)
    for qubit in set(ops_a).intersection(ops_b):
        if ops_a[qubit] != ops_b[qubit]:
            return False
    return True


def summarize_pauli_structure(hamiltonian: Hamiltonian, mapping: str) -> PauliStructureSummary:
    """Summarize mapped Pauli terms for one Hamiltonian."""

    pauli_terms = hamiltonian.get_all_pauli_strings(return_as="tuples")
    non_identity_terms = {
        pauli: coefficient for pauli, coefficient in pauli_terms.items() if len(pauli) > 0
    }
    weights = [_pauli_weight(pauli) for pauli in non_identity_terms]
    weight_histogram = {weight: weights.count(weight) for weight in sorted(set(weights))}
    pauli_counts = {"X": 0, "Y": 0, "Z": 0}
    for pauli in non_identity_terms:
        for _, op in pauli:
            pauli_counts[op] += 1

    anticommute_pairs = 0
    qubit_wise_commuting_pairs = 0
    non_identity_keys = list(non_identity_terms)
    for i, pauli_i in enumerate(non_identity_keys):
        for pauli_j in non_identity_keys[i + 1:]:
            if _pauli_anticommute(pauli_i, pauli_j):
                anticommute_pairs += 1
            if _pauli_qubit_wise_commute(pauli_i, pauli_j):
                qubit_wise_commuting_pairs += 1

    total_pairs = len(non_identity_keys) * (len(non_identity_keys) - 1) // 2
    anticommute_density = anticommute_pairs / total_pairs if total_pairs else 0.0
    commuting_pairs = total_pairs - anticommute_pairs
    commuting_density = commuting_pairs / total_pairs if total_pairs else 0.0
    qubit_wise_commuting_density = (
        qubit_wise_commuting_pairs / total_pairs if total_pairs else 0.0
    )

    return PauliStructureSummary(
        mapping=mapping,
        num_qubits=hamiltonian.num_qubits(),
        num_terms=len(pauli_terms),
        num_non_identity_terms=len(non_identity_terms),
        identity_coefficient=float(np.real(pauli_terms.get(tuple(), 0.0))),
        one_norm=float(sum(abs(coefficient) for coefficient in pauli_terms.values())),
        non_identity_one_norm=float(sum(abs(coefficient) for coefficient in non_identity_terms.values())),
        coefficient_two_norm=float(
            math.sqrt(sum(abs(coefficient) ** 2 for coefficient in pauli_terms.values()))
        ),
        max_pauli_weight=max(weights) if weights else 0,
        mean_pauli_weight=float(np.mean(weights)) if weights else 0.0,
        weight_histogram=weight_histogram,
        x_count=pauli_counts["X"],
        y_count=pauli_counts["Y"],
        z_count=pauli_counts["Z"],
        commuting_pairs=commuting_pairs,
        commuting_density=float(commuting_density),
        qubit_wise_commuting_pairs=qubit_wise_commuting_pairs,
        qubit_wise_commuting_density=float(qubit_wise_commuting_density),
        anticommute_pairs=anticommute_pairs,
        total_pairs=total_pairs,
        anticommute_density=float(anticommute_density),
    )


def summarize_trotter_coefficients(
    hamiltonian: Hamiltonian,
    mapping: str,
    *,
    time_limit: float = 30.0,
    mode: str = "monte_carlo",
    auto_exact: bool = True,
    batch_size: int = 10_000,
    time: float = 1.0,
    energy_error: float | None = None,
    ordering_method: str | None = None,
    random_seed: int | None = None,
    step_counts: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
) -> TrotterCoefficientSummary:
    """
    Compute QHAT's first- and second-order Trotter error coefficients.

    The reported bounds use the standard commutator-scaling forms:
    first order  <= time^2 * C1 / steps
    second order <= time^3 * C2 / steps^2
    """

    ordering_method = _normalize_ordering_method(ordering_method)
    c1, c2 = trotter_error_estimator_fast(
        _grouped_terms(hamiltonian, ordering_method, random_seed),
        time_limit,
        batch_size=batch_size,
        mode=mode,
        auto_exact=auto_exact,
    )

    first_order_steps = None
    second_order_steps = None
    if energy_error is not None:
        if energy_error <= 0:
            raise ValueError("energy_error must be positive when provided.")
        first_order_steps = max(1, math.ceil((time**2) * c1 / energy_error))
        second_order_steps = max(1, math.ceil(math.sqrt((time**3) * c2 / energy_error)))

    first_bounds = {}
    second_bounds = {}
    for steps in step_counts:
        steps = int(steps)
        if steps <= 0:
            continue
        first_bounds[steps] = float((time**2) * c1 / steps)
        second_bounds[steps] = float((time**3) * c2 / (steps**2))

    return TrotterCoefficientSummary(
        mapping=mapping,
        ordering_method=ordering_method,
        random_seed=random_seed if ordering_method == "random" else None,
        c1=float(c1),
        c2=float(c2),
        time=float(time),
        energy_error=float(energy_error) if energy_error is not None else None,
        first_order_steps_estimate=first_order_steps,
        second_order_steps_estimate=second_order_steps,
        first_order_bound_at_steps=first_bounds,
        second_order_bound_at_steps=second_bounds,
    )


def _hamiltonian_matrix(hamiltonian: Hamiltonian) -> np.ndarray:
    pauli_terms = hamiltonian.get_all_pauli_strings(return_as="tuples")
    operator = _terms_to_qubit_operator(pauli_terms)
    return get_sparse_operator(operator, n_qubits=hamiltonian.num_qubits()).toarray()


def _term_matrices(
    hamiltonian: Hamiltonian,
    ordering_method: str | None = None,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    num_qubits = hamiltonian.num_qubits()
    matrices = []
    for pauli_key, coefficient in _ordered_sparse_items(
        hamiltonian,
        ordering_method,
        random_seed,
    ):
        operator = QubitOperator(pauli_key, coefficient)
        matrices.append(get_sparse_operator(operator, n_qubits=num_qubits).toarray())
    return matrices


def _trotter_first_order(term_matrices: list[np.ndarray], time: float, steps: int) -> np.ndarray:
    from scipy.linalg import expm

    dt = time / steps
    dim = term_matrices[0].shape[0]
    step_unitary = np.eye(dim, dtype=complex)
    for term_matrix in term_matrices:
        step_unitary = expm(-1j * dt * term_matrix) @ step_unitary

    unitary = np.eye(dim, dtype=complex)
    for _ in range(steps):
        unitary = step_unitary @ unitary
    return unitary


def _trotter_second_order(term_matrices: list[np.ndarray], time: float, steps: int) -> np.ndarray:
    from scipy.linalg import expm

    dt = time / steps
    dim = term_matrices[0].shape[0]
    forward = np.eye(dim, dtype=complex)
    for term_matrix in term_matrices:
        forward = expm(-1j * (dt / 2) * term_matrix) @ forward

    backward = np.eye(dim, dtype=complex)
    for term_matrix in reversed(term_matrices):
        backward = expm(-1j * (dt / 2) * term_matrix) @ backward

    step_unitary = backward @ forward
    unitary = np.eye(dim, dtype=complex)
    for _ in range(steps):
        unitary = step_unitary @ unitary
    return unitary


def occupation_basis_state(
    num_qubits: int,
    occupied_orbitals: Iterable[int],
    mapping: str,
) -> np.ndarray:
    """
    Build a computational-basis state for a fermionic occupation pattern.

    For JW the occupation vector is already the qubit bit string.  For BK this
    uses the same OpenFermion BK encoder used by the Hamiltonian generator.
    """

    occupation = np.zeros(num_qubits, dtype=int)
    for orbital in occupied_orbitals:
        orbital = int(orbital)
        if orbital < 0 or orbital >= num_qubits:
            raise ValueError(
                f"Occupied orbital {orbital} is outside [0, {num_qubits})."
            )
        occupation[orbital] = 1

    mapping_upper = mapping.upper()
    if "JW" in mapping_upper or "JORDAN" in mapping_upper:
        qubit_bits = occupation
    elif "BK" in mapping_upper or "BRAVYI" in mapping_upper:
        from openfermion.transforms.opconversions.binary_codes import _encoder_bk

        qubit_bits = (_encoder_bk(num_qubits) @ occupation) % 2
    else:
        raise ValueError(
            "Occupation-basis states require mapping to be JW or BK; "
            f"received {mapping!r}."
        )

    basis_index = sum(
        int(qubit_bits[i]) << (num_qubits - 1 - i)
        for i in range(num_qubits)
    )
    state = np.zeros(2 ** num_qubits, dtype=complex)
    state[basis_index] = 1.0
    return state


def code_space_state(
    num_qubits: int,
    kind: str | None,
    *,
    random_seed: int | None = None,
) -> np.ndarray | None:
    """Build a simple qubit-register state for diagnostics."""

    if kind in (None, "", "none"):
        return None
    dim = 2 ** num_qubits
    if kind == "zero":
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        return state
    if kind == "plus":
        return np.full(dim, 1 / math.sqrt(dim), dtype=complex)
    if kind == "random":
        rng = np.random.default_rng(random_seed)
        state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        return state / np.linalg.norm(state)
    raise ValueError(f"Unsupported initial state kind {kind!r}.")


def _resolve_initial_state(
    hamiltonian: Hamiltonian,
    mapping: str,
    initial_state_kind: str | None,
    occupied_orbitals: Iterable[int] | None,
    random_seed: int | None,
) -> tuple[np.ndarray | None, str | None]:
    if occupied_orbitals is not None:
        occupied = tuple(int(orbital) for orbital in occupied_orbitals)
        return (
            occupation_basis_state(hamiltonian.num_qubits(), occupied, mapping),
            f"{mapping} occupation {occupied}",
        )
    state = code_space_state(
        hamiltonian.num_qubits(),
        initial_state_kind,
        random_seed=random_seed,
    )
    if state is None:
        return None, None
    return state, f"{initial_state_kind} code-space state"


def summarize_actual_trotter_errors(
    hamiltonian: Hamiltonian,
    mapping: str,
    *,
    time: float = 1.0,
    step_counts: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
    ordering_method: str | None = None,
    random_seed: int | None = None,
    initial_state: np.ndarray | None = None,
    state_label: str | None = None,
    max_num_qubits: int = 10,
) -> TrotterActualErrorSummary:
    """Compute dense exact-vs-Trotter errors for small systems."""

    from scipy.linalg import expm

    ordering_method = _normalize_ordering_method(ordering_method)
    num_qubits = hamiltonian.num_qubits()
    if max_num_qubits is not None and num_qubits > max_num_qubits:
        raise ValueError(
            "Dense actual-error comparison would build matrices of size "
            f"{2 ** num_qubits} x {2 ** num_qubits}. "
            f"Requested {num_qubits} qubits, but max_num_qubits is {max_num_qubits}."
        )

    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=complex).ravel()
        dim = 2 ** num_qubits
        if initial_state.shape != (dim,):
            raise ValueError(
                f"initial_state has shape {initial_state.shape}, expected {(dim,)}."
            )
        norm = np.linalg.norm(initial_state)
        if norm == 0:
            raise ValueError("initial_state cannot be the zero vector.")
        initial_state = initial_state / norm

    exact = expm(-1j * time * _hamiltonian_matrix(hamiltonian))
    terms = _term_matrices(hamiltonian, ordering_method, random_seed)

    first_errors = {}
    second_errors = {}
    first_frobenius_errors = {}
    second_frobenius_errors = {}
    first_state_errors = {} if initial_state is not None else None
    second_state_errors = {} if initial_state is not None else None
    first_state_infidelities = {} if initial_state is not None else None
    second_state_infidelities = {} if initial_state is not None else None
    exact_state = exact @ initial_state if initial_state is not None else None
    for steps in step_counts:
        steps = int(steps)
        if steps <= 0:
            continue
        first = _trotter_first_order(terms, time, steps)
        second = _trotter_second_order(terms, time, steps)
        first_errors[steps] = float(np.linalg.norm(first - exact, ord=2))
        second_errors[steps] = float(np.linalg.norm(second - exact, ord=2))
        first_frobenius_errors[steps] = float(np.linalg.norm(first - exact, ord="fro"))
        second_frobenius_errors[steps] = float(np.linalg.norm(second - exact, ord="fro"))

        if initial_state is not None:
            first_state = first @ initial_state
            second_state = second @ initial_state
            first_state_errors[steps] = float(np.linalg.norm(first_state - exact_state))
            second_state_errors[steps] = float(np.linalg.norm(second_state - exact_state))
            first_overlap = np.vdot(exact_state, first_state)
            second_overlap = np.vdot(exact_state, second_state)
            first_state_infidelities[steps] = float(
                max(0.0, 1.0 - abs(first_overlap) ** 2)
            )
            second_state_infidelities[steps] = float(
                max(0.0, 1.0 - abs(second_overlap) ** 2)
            )

    return TrotterActualErrorSummary(
        mapping=mapping,
        ordering_method=ordering_method,
        random_seed=random_seed if ordering_method == "random" else None,
        state_label=state_label,
        time=float(time),
        first_order_operator_error=first_errors,
        second_order_operator_error=second_errors,
        first_order_frobenius_error=first_frobenius_errors,
        second_order_frobenius_error=second_frobenius_errors,
        first_order_state_error=first_state_errors,
        second_order_state_error=second_state_errors,
        first_order_state_infidelity=first_state_infidelities,
        second_order_state_infidelity=second_state_infidelities,
    )


def _compare_pauli_spectra(
    left: Hamiltonian,
    right: Hamiltonian,
    *,
    max_num_qubits: int = 12,
    atol: float = 1.0e-10,
) -> float:
    if left.num_qubits() != right.num_qubits():
        raise ValueError(
            f"Pauli Hamiltonians use different qubit counts: "
            f"{left.num_qubits()} and {right.num_qubits()}."
        )
    num_qubits = left.num_qubits()
    if max_num_qubits is not None and num_qubits > max_num_qubits:
        raise ValueError(
            "Dense Pauli spectral comparison would build matrices of size "
            f"{2 ** num_qubits} x {2 ** num_qubits}. "
            f"Requested {num_qubits} qubits, but max_num_qubits is {max_num_qubits}."
        )

    left_op = _terms_to_qubit_operator(left.get_all_pauli_strings(return_as="tuples"))
    right_op = _terms_to_qubit_operator(right.get_all_pauli_strings(return_as="tuples"))
    left_matrix = get_sparse_operator(left_op, n_qubits=num_qubits).toarray()
    right_matrix = get_sparse_operator(right_op, n_qubits=num_qubits).toarray()

    if not np.allclose(left_matrix, left_matrix.conjugate().T, atol=atol, rtol=0.0):
        raise ValueError("Left Pauli Hamiltonian is not Hermitian.")
    if not np.allclose(right_matrix, right_matrix.conjugate().T, atol=atol, rtol=0.0):
        raise ValueError("Right Pauli Hamiltonian is not Hermitian.")

    left_eig = np.sort(np.linalg.eigvalsh(left_matrix))
    right_eig = np.sort(np.linalg.eigvalsh(right_matrix))
    return float(np.max(np.abs(left_eig - right_eig)))


def _actual_summary_if_requested(
    hamiltonian: Hamiltonian,
    mapping: str,
    *,
    actual_errors: bool,
    time: float,
    step_counts: Iterable[int],
    ordering_method: str | None,
    random_seed: int | None,
    initial_state_kind: str | None,
    occupied_orbitals: Iterable[int] | None,
    actual_max_num_qubits: int,
) -> TrotterActualErrorSummary | None:
    if not actual_errors:
        return None
    initial_state, state_label = _resolve_initial_state(
        hamiltonian,
        mapping,
        initial_state_kind,
        occupied_orbitals,
        random_seed,
    )
    return summarize_actual_trotter_errors(
        hamiltonian,
        mapping,
        time=time,
        step_counts=step_counts,
        ordering_method=ordering_method,
        random_seed=random_seed,
        initial_state=initial_state,
        state_label=state_label,
        max_num_qubits=actual_max_num_qubits,
    )


def compare_fermionic_hamiltonian(
    fermionic_hamiltonian: FermionOperator | InteractionOperator,
    *,
    label: str = "fermionic Hamiltonian",
    num_qubits: int | None = None,
    time_limit: float = 30.0,
    mode: str = "monte_carlo",
    auto_exact: bool = True,
    batch_size: int = 10_000,
    time: float = 1.0,
    energy_error: float | None = None,
    ordering_method: str | None = None,
    random_seed: int | None = None,
    step_counts: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
    actual_errors: bool = False,
    initial_state_kind: str | None = None,
    occupied_orbitals: Iterable[int] | None = None,
    actual_max_num_qubits: int = 10,
    max_num_qubits: int = 12,
) -> MappingReport:
    """Compare JW and BK from one in-memory fermionic Hamiltonian."""

    resolved_num_qubits = _infer_fermionic_num_qubits(fermionic_hamiltonian, num_qubits)
    jw_hamiltonian = _mapped_qubit_hamiltonian(
        fermionic_hamiltonian,
        "JW",
        num_qubits=resolved_num_qubits,
    )
    bk_hamiltonian = _mapped_qubit_hamiltonian(
        fermionic_hamiltonian,
        "BK",
        num_qubits=resolved_num_qubits,
    )
    spectrum = compare_jordan_wigner_bravyi_kitaev_spectra(
        fermionic_hamiltonian,
        num_qubits=resolved_num_qubits,
        max_num_qubits=max_num_qubits,
    )

    return MappingReport(
        label=label,
        jw_structure=summarize_pauli_structure(jw_hamiltonian, "JW"),
        bk_structure=summarize_pauli_structure(bk_hamiltonian, "BK"),
        jw_trotter=summarize_trotter_coefficients(
            jw_hamiltonian,
            "JW",
            time_limit=time_limit,
            mode=mode,
            auto_exact=auto_exact,
            batch_size=batch_size,
            time=time,
            energy_error=energy_error,
            ordering_method=ordering_method,
            random_seed=random_seed,
            step_counts=step_counts,
        ),
        bk_trotter=summarize_trotter_coefficients(
            bk_hamiltonian,
            "BK",
            time_limit=time_limit,
            mode=mode,
            auto_exact=auto_exact,
            batch_size=batch_size,
            time=time,
            energy_error=energy_error,
            ordering_method=ordering_method,
            random_seed=random_seed,
            step_counts=step_counts,
        ),
        jw_actual_errors=_actual_summary_if_requested(
            jw_hamiltonian,
            "JW",
            actual_errors=actual_errors,
            time=time,
            step_counts=step_counts,
            ordering_method=ordering_method,
            random_seed=random_seed,
            initial_state_kind=initial_state_kind,
            occupied_orbitals=occupied_orbitals,
            actual_max_num_qubits=actual_max_num_qubits,
        ),
        bk_actual_errors=_actual_summary_if_requested(
            bk_hamiltonian,
            "BK",
            actual_errors=actual_errors,
            time=time,
            step_counts=step_counts,
            ordering_method=ordering_method,
            random_seed=random_seed,
            initial_state_kind=initial_state_kind,
            occupied_orbitals=occupied_orbitals,
            actual_max_num_qubits=actual_max_num_qubits,
        ),
        spectrum=spectrum,
        pauli_spectrum_max_abs_difference=None,
        notes=(
            "JW and BK were generated from the same in-memory fermionic Hamiltonian.",
            "Trotter diagnostics compare the product formulas after mapping; "
            "different Pauli supports and commutation graphs can change the error.",
        ),
    )


def compare_second_quantized_file(
    filename: str,
    *,
    time_limit: float = 30.0,
    mode: str = "monte_carlo",
    auto_exact: bool = True,
    batch_size: int = 10_000,
    time: float = 1.0,
    energy_error: float | None = None,
    ordering_method: str | None = None,
    random_seed: int | None = None,
    step_counts: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
    actual_errors: bool = False,
    initial_state_kind: str | None = None,
    occupied_orbitals: Iterable[int] | None = None,
    actual_max_num_qubits: int = 10,
    max_num_qubits: int = 12,
) -> MappingReport:
    """Compare JW and BK from the same second-quantized Hamiltonian file."""

    jw_hamiltonian = _load_second_quantized(filename, "JW")
    bk_hamiltonian = _load_second_quantized(filename, "BK")

    spectrum = compare_jordan_wigner_bravyi_kitaev_spectra(
        jw_hamiltonian,
        max_num_qubits=max_num_qubits,
    )

    return MappingReport(
        label=str(filename),
        jw_structure=summarize_pauli_structure(jw_hamiltonian, "JW"),
        bk_structure=summarize_pauli_structure(bk_hamiltonian, "BK"),
        jw_trotter=summarize_trotter_coefficients(
            jw_hamiltonian,
            "JW",
            time_limit=time_limit,
            mode=mode,
            auto_exact=auto_exact,
            batch_size=batch_size,
            time=time,
            energy_error=energy_error,
            ordering_method=ordering_method,
            random_seed=random_seed,
            step_counts=step_counts,
        ),
        bk_trotter=summarize_trotter_coefficients(
            bk_hamiltonian,
            "BK",
            time_limit=time_limit,
            mode=mode,
            auto_exact=auto_exact,
            batch_size=batch_size,
            time=time,
            energy_error=energy_error,
            ordering_method=ordering_method,
            random_seed=random_seed,
            step_counts=step_counts,
        ),
        jw_actual_errors=_actual_summary_if_requested(
            jw_hamiltonian,
            "JW",
            actual_errors=actual_errors,
            time=time,
            step_counts=step_counts,
            ordering_method=ordering_method,
            random_seed=random_seed,
            initial_state_kind=initial_state_kind,
            occupied_orbitals=occupied_orbitals,
            actual_max_num_qubits=actual_max_num_qubits,
        ),
        bk_actual_errors=_actual_summary_if_requested(
            bk_hamiltonian,
            "BK",
            actual_errors=actual_errors,
            time=time,
            step_counts=step_counts,
            ordering_method=ordering_method,
            random_seed=random_seed,
            initial_state_kind=initial_state_kind,
            occupied_orbitals=occupied_orbitals,
            actual_max_num_qubits=actual_max_num_qubits,
        ),
        spectrum=spectrum,
        pauli_spectrum_max_abs_difference=None,
        notes=(
            "JW and BK were generated from the same second-quantized Hamiltonian.",
            "Trotter coefficients depend on mapped Pauli strings and their ordering, "
            "so they can differ even when spectra match.",
        ),
    )


def compare_pauli_files(
    jw_filename: str,
    bk_filename: str,
    *,
    time_limit: float = 30.0,
    mode: str = "monte_carlo",
    auto_exact: bool = True,
    batch_size: int = 10_000,
    time: float = 1.0,
    energy_error: float | None = None,
    ordering_method: str | None = None,
    random_seed: int | None = None,
    step_counts: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
    actual_errors: bool = False,
    initial_state_kind: str | None = None,
    occupied_orbitals: Iterable[int] | None = None,
    actual_max_num_qubits: int = 10,
    max_num_qubits: int = 12,
) -> MappingReport:
    """Compare already-mapped JW and BK Pauli Hamiltonian files."""

    jw_hamiltonian = _load_pauli(jw_filename)
    bk_hamiltonian = _load_pauli(bk_filename)
    max_abs_difference = _compare_pauli_spectra(
        jw_hamiltonian,
        bk_hamiltonian,
        max_num_qubits=max_num_qubits,
    )

    return MappingReport(
        label=f"{jw_filename} vs {bk_filename}",
        jw_structure=summarize_pauli_structure(jw_hamiltonian, "JW"),
        bk_structure=summarize_pauli_structure(bk_hamiltonian, "BK"),
        jw_trotter=summarize_trotter_coefficients(
            jw_hamiltonian,
            "JW",
            time_limit=time_limit,
            mode=mode,
            auto_exact=auto_exact,
            batch_size=batch_size,
            time=time,
            energy_error=energy_error,
            ordering_method=ordering_method,
            random_seed=random_seed,
            step_counts=step_counts,
        ),
        bk_trotter=summarize_trotter_coefficients(
            bk_hamiltonian,
            "BK",
            time_limit=time_limit,
            mode=mode,
            auto_exact=auto_exact,
            batch_size=batch_size,
            time=time,
            energy_error=energy_error,
            ordering_method=ordering_method,
            random_seed=random_seed,
            step_counts=step_counts,
        ),
        jw_actual_errors=_actual_summary_if_requested(
            jw_hamiltonian,
            "JW",
            actual_errors=actual_errors,
            time=time,
            step_counts=step_counts,
            ordering_method=ordering_method,
            random_seed=random_seed,
            initial_state_kind=initial_state_kind,
            occupied_orbitals=occupied_orbitals,
            actual_max_num_qubits=actual_max_num_qubits,
        ),
        bk_actual_errors=_actual_summary_if_requested(
            bk_hamiltonian,
            "BK",
            actual_errors=actual_errors,
            time=time,
            step_counts=step_counts,
            ordering_method=ordering_method,
            random_seed=random_seed,
            initial_state_kind=initial_state_kind,
            occupied_orbitals=occupied_orbitals,
            actual_max_num_qubits=actual_max_num_qubits,
        ),
        spectrum=None,
        pauli_spectrum_max_abs_difference=max_abs_difference,
        notes=(
            "JW and BK were loaded from separate Pauli files.",
            "This verifies spectra of the supplied qubit Hamiltonians, not the "
            "fermion-to-qubit transformation itself.",
        ),
    )


def inspect_pauli_file(
    filename: str,
    *,
    label: str = "Pauli",
    time_limit: float = 30.0,
    mode: str = "monte_carlo",
    auto_exact: bool = True,
    batch_size: int = 10_000,
    time: float = 1.0,
    energy_error: float | None = None,
    ordering_method: str | None = None,
    random_seed: int | None = None,
    step_counts: Iterable[int] = (1, 2, 5, 10, 20, 50, 100),
    actual_errors: bool = False,
    initial_state_kind: str | None = None,
    occupied_orbitals: Iterable[int] | None = None,
    actual_max_num_qubits: int = 10,
) -> MappingReport:
    """Inspect one mapped Pauli file when no counterpart mapping is available."""

    hamiltonian = _load_pauli(filename)
    return MappingReport(
        label=str(filename),
        jw_structure=summarize_pauli_structure(hamiltonian, label),
        bk_structure=None,
        jw_trotter=summarize_trotter_coefficients(
            hamiltonian,
            label,
            time_limit=time_limit,
            mode=mode,
            auto_exact=auto_exact,
            batch_size=batch_size,
            time=time,
            energy_error=energy_error,
            ordering_method=ordering_method,
            random_seed=random_seed,
            step_counts=step_counts,
        ),
        bk_trotter=None,
        jw_actual_errors=_actual_summary_if_requested(
            hamiltonian,
            label,
            actual_errors=actual_errors,
            time=time,
            step_counts=step_counts,
            ordering_method=ordering_method,
            random_seed=random_seed,
            initial_state_kind=initial_state_kind,
            occupied_orbitals=occupied_orbitals,
            actual_max_num_qubits=actual_max_num_qubits,
        ),
        bk_actual_errors=None,
        spectrum=None,
        pauli_spectrum_max_abs_difference=None,
        notes=(
            "Only one mapped Pauli Hamiltonian was supplied.",
            "A true JW/BK comparison needs the original second-quantized Hamiltonian "
            "or both mapped Pauli files.",
        ),
    )


def _format_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6e}"


def _print_structure(summary: PauliStructureSummary):
    print(f"  [{summary.mapping}]")
    print(f"    qubits                    : {summary.num_qubits}")
    print(f"    terms                     : {summary.num_terms}")
    print(f"    non-identity terms        : {summary.num_non_identity_terms}")
    print(f"    identity coefficient      : {_format_float(summary.identity_coefficient)}")
    print(f"    one-norm                  : {_format_float(summary.one_norm)}")
    print(f"    non-identity one-norm     : {_format_float(summary.non_identity_one_norm)}")
    print(f"    coefficient two-norm      : {_format_float(summary.coefficient_two_norm)}")
    print(f"    max Pauli weight          : {summary.max_pauli_weight}")
    print(f"    mean Pauli weight         : {summary.mean_pauli_weight:.3f}")
    print(f"    weight histogram          : {summary.weight_histogram}")
    print(
        "    Pauli letter counts       : "
        f"X={summary.x_count}, Y={summary.y_count}, Z={summary.z_count}"
    )
    print(
        f"    commuting pairs           : {summary.commuting_pairs}/"
        f"{summary.total_pairs} ({summary.commuting_density:.3%})"
    )
    print(
        f"    qubit-wise commuting      : {summary.qubit_wise_commuting_pairs}/"
        f"{summary.total_pairs} ({summary.qubit_wise_commuting_density:.3%})"
    )
    print(
        f"    anticommuting pairs       : {summary.anticommute_pairs}/"
        f"{summary.total_pairs} ({summary.anticommute_density:.3%})"
    )


def _print_trotter(summary: TrotterCoefficientSummary):
    print(f"  [{summary.mapping}]")
    print(f"    ordering                  : {_ordering_label(summary.ordering_method)}")
    if summary.random_seed is not None:
        print(f"    random seed               : {summary.random_seed}")
    print(f"    C1                        : {_format_float(summary.c1)}")
    print(f"    C2                        : {_format_float(summary.c2)}")
    if summary.energy_error is not None:
        print(f"    target energy error       : {_format_float(summary.energy_error)}")
        print(f"    1st-order step estimate   : {summary.first_order_steps_estimate}")
        print(f"    2nd-order step estimate   : {summary.second_order_steps_estimate}")
    print("    fixed-step bounds:")
    for steps in summary.first_order_bound_at_steps:
        first = summary.first_order_bound_at_steps[steps]
        second = summary.second_order_bound_at_steps[steps]
        print(f"      steps={steps:<4d} first={first:.6e} second={second:.6e}")


def _print_actual_errors(summary: TrotterActualErrorSummary):
    print(f"  [{summary.mapping}]")
    print(f"    ordering                  : {_ordering_label(summary.ordering_method)}")
    if summary.random_seed is not None:
        print(f"    random seed               : {summary.random_seed}")
    if summary.state_label is not None:
        print(f"    state                     : {summary.state_label}")
    for steps in summary.first_order_operator_error:
        first = summary.first_order_operator_error[steps]
        second = summary.second_order_operator_error[steps]
        first_fro = summary.first_order_frobenius_error[steps]
        second_fro = summary.second_order_frobenius_error[steps]
        print(
            f"    steps={steps:<4d} "
            f"spec(first={first:.6e}, second={second:.6e}) "
            f"fro(first={first_fro:.6e}, second={second_fro:.6e})"
        )
        if summary.first_order_state_infidelity is not None:
            first_infidelity = summary.first_order_state_infidelity[steps]
            second_infidelity = summary.second_order_state_infidelity[steps]
            print(
                f"                 "
                f"state infidelity(first={first_infidelity:.6e}, "
                f"second={second_infidelity:.6e})"
            )


def print_report(report: MappingReport):
    """Print a human-readable comparison report."""

    print("=" * 88)
    print(f"Mapping comparison: {report.label}")
    print("=" * 88)

    if report.spectrum is not None:
        print("Spectrum:")
        print(f"  qubits                    : {report.spectrum.num_qubits}")
        print(f"  isospectral               : {report.spectrum.is_isospectral}")
        print(f"  max abs eigenvalue diff   : {_format_float(report.spectrum.max_abs_difference)}")
    elif report.pauli_spectrum_max_abs_difference is not None:
        print("Spectrum:")
        print(f"  isospectral               : {report.is_isospectral}")
        print(
            "  max abs eigenvalue diff   : "
            f"{_format_float(report.pauli_spectrum_max_abs_difference)}"
        )
    else:
        print("Spectrum:")
        print("  n/a                       : only one mapped Pauli Hamiltonian supplied")

    print("\nPauli structure:")
    _print_structure(report.jw_structure)
    if report.bk_structure is not None:
        _print_structure(report.bk_structure)

    print("\nTrotter coefficients:")
    _print_trotter(report.jw_trotter)
    if report.bk_trotter is not None:
        _print_trotter(report.bk_trotter)

    if report.jw_actual_errors is not None:
        print("\nActual dense Trotter operator errors:")
        _print_actual_errors(report.jw_actual_errors)
        if report.bk_actual_errors is not None:
            _print_actual_errors(report.bk_actual_errors)

    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"  - {note}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def two_mode_toy_fermionic_hamiltonian() -> FermionOperator:
    """Small Hermitian two-mode fermionic Hamiltonian for smoke tests."""

    hamiltonian = FermionOperator("", 0.4)
    hamiltonian += FermionOperator("0^ 0", 0.7)
    hamiltonian += FermionOperator("1^ 1", -0.2)
    hamiltonian += FermionOperator("0^ 1", 0.3)
    hamiltonian += FermionOperator("1^ 0", 0.3)
    hamiltonian += FermionOperator("0^ 0 1^ 1", 0.5)
    return hamiltonian


def _synthetic_two_mode(args):
    report = compare_fermionic_hamiltonian(
        two_mode_toy_fermionic_hamiltonian(),
        label="two-mode synthetic fermionic Hamiltonian",
        num_qubits=2,
        max_num_qubits=args.max_num_qubits,
        **_comparison_kwargs(args),
    )
    _emit(report, args.json)


def _demo(args):
    root = _repo_root()
    be_h = root / "analysis" / "examples" / "Be-H_1.30_sto-6g_as-003-003.tensors.npz"
    h2 = root / "analysis" / "tests" / "hamlib_h2_sto6g.hdf5"
    lili = root / "julia_trotter" / "Li-Li_jw" / "Li-Li_2.90_hgbs-5_as-002-002_jw.dat"

    reports = [
        compare_second_quantized_file(
            str(be_h),
            max_num_qubits=args.max_num_qubits,
            **_comparison_kwargs(args),
        ),
        inspect_pauli_file(str(h2), label="H2 Pauli", **_comparison_kwargs(args)),
        inspect_pauli_file(str(lili), label="Li-Li JW", **_comparison_kwargs(args)),
    ]

    if args.json:
        print(json.dumps([report.as_dict() for report in reports], indent=2))
    else:
        for idx, report in enumerate(reports):
            if idx:
                print()
            print_report(report)


def _comparison_kwargs(args) -> dict:
    return {
        "time_limit": args.time_limit,
        "mode": args.mode,
        "auto_exact": args.auto_exact,
        "batch_size": args.batch_size,
        "time": args.time,
        "energy_error": args.energy_error,
        "ordering_method": _normalize_ordering_method(args.ordering_method),
        "random_seed": args.random_seed,
        "step_counts": args.step_counts,
        "actual_errors": args.actual_errors,
        "initial_state_kind": args.initial_state,
        "occupied_orbitals": args.occupied_orbitals,
        "actual_max_num_qubits": args.actual_max_num_qubits,
    }


def _parse_step_counts(value: str) -> tuple[int, ...]:
    try:
        step_counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --step-counts value {value!r}; use comma-separated integers."
        ) from exc
    if not step_counts or any(step <= 0 for step in step_counts):
        raise argparse.ArgumentTypeError("--step-counts must contain positive integers.")
    return step_counts


def _parse_occupied_orbitals(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip().lower() in ("", "none"):
        return None
    try:
        occupied = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --occupied-orbitals value {value!r}; use comma-separated integers."
        ) from exc
    if any(orbital < 0 for orbital in occupied):
        raise argparse.ArgumentTypeError("--occupied-orbitals cannot contain negative indices.")
    return occupied


def _add_common_args(parser):
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mode", choices=["monte_carlo", "exact"], default="monte_carlo")
    parser.set_defaults(auto_exact=True)
    parser.add_argument("--auto-exact", dest="auto_exact", action="store_true")
    parser.add_argument("--no-auto-exact", dest="auto_exact", action="store_false")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--energy-error", type=float, default=None)
    parser.add_argument(
        "--ordering-method",
        choices=["native", "magnitude", "lexicographical", "random"],
        default=None,
        help="Optional Pauli-term ordering to use before Trotter comparison.",
    )
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--step-counts",
        type=_parse_step_counts,
        default=(1, 2, 5, 10, 20, 50, 100),
    )
    parser.add_argument(
        "--actual-errors",
        action="store_true",
        help="Also compute dense exact-vs-Trotter operator errors for small systems.",
    )
    parser.add_argument(
        "--initial-state",
        choices=["none", "zero", "plus", "random"],
        default="none",
        help="Optional code-space state for dense state-vector error diagnostics.",
    )
    parser.add_argument(
        "--occupied-orbitals",
        type=_parse_occupied_orbitals,
        default=None,
        help=(
            "Comma-separated occupied spin-orbital indices for a physical "
            "occupation-basis state. Overrides --initial-state."
        ),
    )
    parser.add_argument("--actual-max-num-qubits", type=int, default=10)
    parser.add_argument("--max-num-qubits", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")


def _emit(report: MappingReport, as_json: bool):
    if as_json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print_report(report)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Compare JW and BK spectra and Trotter error coefficients."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    second = subparsers.add_parser(
        "second-quantized",
        help="Generate JW and BK from one second-quantized Hamiltonian file.",
    )
    second.add_argument("filename")
    _add_common_args(second)

    pair = subparsers.add_parser(
        "pauli-pair",
        help="Compare two already-mapped Pauli Hamiltonian files.",
    )
    pair.add_argument("jw_filename")
    pair.add_argument("bk_filename")
    _add_common_args(pair)

    single = subparsers.add_parser(
        "inspect-pauli",
        help="Inspect one Pauli Hamiltonian when no counterpart mapping is available.",
    )
    single.add_argument("filename")
    single.add_argument("--label", default="Pauli")
    _add_common_args(single)

    demo = subparsers.add_parser(
        "demo",
        help="Run the checked-in Be-H comparison and inspect checked-in H2/Li-Li Pauli files.",
    )
    _add_common_args(demo)

    synthetic = subparsers.add_parser(
        "synthetic-two-mode",
        help="Run a dependency-light two-mode fermionic JW/BK comparison.",
    )
    _add_common_args(synthetic)

    args = parser.parse_args(argv)

    if args.command == "second-quantized":
        report = compare_second_quantized_file(
            args.filename,
            max_num_qubits=args.max_num_qubits,
            **_comparison_kwargs(args),
        )
        _emit(report, args.json)
    elif args.command == "pauli-pair":
        report = compare_pauli_files(
            args.jw_filename,
            args.bk_filename,
            max_num_qubits=args.max_num_qubits,
            **_comparison_kwargs(args),
        )
        _emit(report, args.json)
    elif args.command == "inspect-pauli":
        report = inspect_pauli_file(
            args.filename,
            label=args.label,
            **_comparison_kwargs(args),
        )
        _emit(report, args.json)
    elif args.command == "demo":
        _demo(args)
    elif args.command == "synthetic-two-mode":
        _synthetic_two_mode(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
