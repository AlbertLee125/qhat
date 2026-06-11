import numpy as np
import pytest
from openfermion import FermionOperator, InteractionOperator

from qhat.analysis.hamiltonian import Hamiltonian
from qhat.analysis.mapping_isomorphism import (
    assert_jordan_wigner_bravyi_kitaev_isospectral,
    find_jw_bk_eigenvalue_match,
)


def _two_mode_fermion_hamiltonian():
    hamiltonian = FermionOperator("", 0.4)
    hamiltonian += FermionOperator("0^ 0", 0.7)
    hamiltonian += FermionOperator("1^ 1", -0.2)
    hamiltonian += FermionOperator("0^ 1", 0.3)
    hamiltonian += FermionOperator("1^ 0", 0.3)
    hamiltonian += FermionOperator("0^ 0 1^ 1", 0.5)
    return hamiltonian


def test_find_jw_bk_eigenvalue_match_returns_equal_spectra():
    comparison = find_jw_bk_eigenvalue_match(_two_mode_fermion_hamiltonian())

    assert comparison.num_qubits == 2
    assert comparison.is_isospectral
    assert comparison.max_abs_difference < 1.0e-10
    np.testing.assert_allclose(
        comparison.jw_eigenvalues,
        comparison.bk_eigenvalues,
        atol=1.0e-10,
        rtol=1.0e-8,
    )


def test_assertion_helper_accepts_qhat_hamiltonian_wrapper():
    one_body = np.array([
        [0.5, 0.125],
        [0.125, -0.25],
    ])
    two_body = np.zeros((2, 2, 2, 2))
    interaction_operator = InteractionOperator(0.1, one_body, two_body)
    wrapped_hamiltonian = Hamiltonian(interaction_operator)

    comparison = assert_jordan_wigner_bravyi_kitaev_isospectral(wrapped_hamiltonian)

    assert comparison.num_qubits == 2
    assert comparison.is_isospectral


def test_identity_only_operator_can_use_explicit_code_space_size():
    comparison = find_jw_bk_eigenvalue_match(
        FermionOperator("", 2.5),
        num_qubits=3,
    )

    assert comparison.num_qubits == 3
    assert comparison.is_isospectral
    assert len(comparison.jw_eigenvalues) == 8
    np.testing.assert_allclose(comparison.jw_eigenvalues, np.full(8, 2.5))


def test_dense_comparison_rejects_too_many_qubits():
    with pytest.raises(ValueError, match="max_num_qubits"):
        find_jw_bk_eigenvalue_match(
            FermionOperator("4^ 4", 1.0),
            max_num_qubits=3,
        )


def test_non_hermitian_operator_raises_clear_error():
    with pytest.raises(ValueError, match="not Hermitian"):
        find_jw_bk_eigenvalue_match(FermionOperator("0^ 1", 1.0))
