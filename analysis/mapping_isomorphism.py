
"""
Spectral checks for fermion-to-qubit mapping isomorphisms.
Jordan-Wigner and Bravyi-Kitaev encode the same fermionic operator algebra in
different qubit bases.  For a finite Hamiltonian, a direct way to check that the
two encodings represent the same physics is to compare the eigenvalues of the
two mapped qubit Hamiltonians.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from openfermion import (
    FermionOperator,
    InteractionOperator,
    bravyi_kitaev,
    count_qubits,
    jordan_wigner,
)
from openfermion.linalg import get_sparse_operator

from qhat.analysis.hamiltonian import Hamiltonian


@dataclass(frozen=True)
class FermionMappingSpectrumComparison:
    """Eigenvalue comparison between two fermion-to-qubit mappings."""

    num_qubits: int
    jw_eigenvalues: np.ndarray
    bk_eigenvalues: np.ndarray
    max_abs_difference: float
    atol: float
    rtol: float

    @property
    def is_isospectral(self) -> bool:
        """Return True when the two spectra agree within the configured tolerance."""
        return bool(np.allclose(
            self.jw_eigenvalues,
            self.bk_eigenvalues,
            atol=self.atol,
            rtol=self.rtol,
        ))

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable summary of the spectral comparison."""
        return {
            "num_qubits": self.num_qubits,
            "is_isospectral": self.is_isospectral,
            "max_abs_difference": self.max_abs_difference,
            "atol": self.atol,
            "rtol": self.rtol,
            "jw_eigenvalues": self.jw_eigenvalues.tolist(),
            "bk_eigenvalues": self.bk_eigenvalues.tolist(),
        }


def _unwrap_fermionic_operator(hamiltonian):
    if isinstance(hamiltonian, Hamiltonian):
        return hamiltonian.get_core_operator(), hamiltonian.num_qubits()
    return hamiltonian, None


def _infer_num_qubits(fermionic_operator, wrapped_num_qubits, num_qubits):
    if num_qubits is not None:
        if num_qubits < 0:
            raise ValueError("num_qubits must be non-negative.")
        return int(num_qubits)
    if wrapped_num_qubits is not None:
        return int(wrapped_num_qubits)
    if hasattr(fermionic_operator, "n_qubits"):
        return int(fermionic_operator.n_qubits)
    return int(count_qubits(fermionic_operator))


def _validate_fermionic_operator(fermionic_operator):
    if isinstance(fermionic_operator, (FermionOperator, InteractionOperator)):
        return
    raise TypeError(
        "JW/BK spectral comparison requires an OpenFermion FermionOperator, "
        "InteractionOperator, or qhat.analysis.hamiltonian.Hamiltonian wrapping "
        "one of those operator types."
    )


def _sorted_hermitian_eigenvalues(qubit_operator, num_qubits, hermitian_atol):
    matrix = get_sparse_operator(qubit_operator, n_qubits=num_qubits).toarray()
    if not np.allclose(matrix, matrix.conjugate().T, atol=hermitian_atol, rtol=0.0):
        skew = matrix - matrix.conjugate().T
        skew_norm = float(np.max(np.abs(skew)))
        raise ValueError(
            "Encoded qubit operator is not Hermitian; "
            f"max |H - H^dagger| = {skew_norm}."
        )
    return np.sort(np.linalg.eigvalsh(matrix))


def compare_jordan_wigner_bravyi_kitaev_spectra(
    hamiltonian,
    *,
    num_qubits=None,
    max_num_qubits=12,
    atol=1.0e-10,
    rtol=1.0e-8,
    hermitian_atol=1.0e-10,
):
    """
    Compare JW and BK spectra for the same fermionic Hamiltonian.
    Args:
        hamiltonian: OpenFermion ``FermionOperator``/``InteractionOperator`` or
            a QHAT ``Hamiltonian`` wrapping an ``InteractionOperator``.
        num_qubits: Optional qubit count.  Supplying this is useful for
            identity-only ``FermionOperator`` instances, where OpenFermion cannot
            infer the intended register size from operator terms.
        max_num_qubits: Refuse dense diagonalization above this number of
            qubits.  The dense matrices have dimension ``2 ** num_qubits``.
        atol: Absolute tolerance used when deciding whether spectra match.
        rtol: Relative tolerance used when deciding whether spectra match.
        hermitian_atol: Absolute tolerance used when checking Hermiticity before
            diagonalization.
    Returns:
        ``FermionMappingSpectrumComparison`` with the sorted eigenvalues and the
        maximum absolute eigenvalue difference.
    """
    fermionic_operator, wrapped_num_qubits = _unwrap_fermionic_operator(hamiltonian)
    _validate_fermionic_operator(fermionic_operator)

    resolved_num_qubits = _infer_num_qubits(
        fermionic_operator,
        wrapped_num_qubits,
        num_qubits,
    )
    if max_num_qubits is not None and resolved_num_qubits > max_num_qubits:
        raise ValueError(
            "Dense JW/BK spectral comparison would build matrices of size "
            f"{2 ** resolved_num_qubits} x {2 ** resolved_num_qubits}. "
            f"Requested {resolved_num_qubits} qubits, but max_num_qubits is "
            f"{max_num_qubits}."
        )

    jw_operator = jordan_wigner(fermionic_operator)
    bk_operator = bravyi_kitaev(fermionic_operator, n_qubits=resolved_num_qubits)

    jw_eigenvalues = _sorted_hermitian_eigenvalues(
        jw_operator,
        resolved_num_qubits,
        hermitian_atol,
    )
    bk_eigenvalues = _sorted_hermitian_eigenvalues(
        bk_operator,
        resolved_num_qubits,
        hermitian_atol,
    )
    max_abs_difference = float(np.max(np.abs(jw_eigenvalues - bk_eigenvalues)))

    return FermionMappingSpectrumComparison(
        num_qubits=resolved_num_qubits,
        jw_eigenvalues=jw_eigenvalues,
        bk_eigenvalues=bk_eigenvalues,
        max_abs_difference=max_abs_difference,
        atol=atol,
        rtol=rtol,
    )


def find_jw_bk_eigenvalue_match(*args, **kwargs):
    """Alias for ``compare_jordan_wigner_bravyi_kitaev_spectra``."""
    return compare_jordan_wigner_bravyi_kitaev_spectra(*args, **kwargs)


def assert_jordan_wigner_bravyi_kitaev_isospectral(*args, **kwargs):
    """
    Assert that JW and BK encodings have the same spectrum.
    Returns the comparison result on success so callers can also inspect or
    report the eigenvalues.
    """
    comparison = compare_jordan_wigner_bravyi_kitaev_spectra(*args, **kwargs)
    if not comparison.is_isospectral:
        raise AssertionError(
            "Jordan-Wigner and Bravyi-Kitaev spectra do not match: "
            f"max_abs_difference={comparison.max_abs_difference}, "
            f"atol={comparison.atol}, rtol={comparison.rtol}."
        )
    return comparison