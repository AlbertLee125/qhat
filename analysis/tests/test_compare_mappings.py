import numpy as np

from qhat.analysis.compare_mappings import (
    compare_fermionic_hamiltonian,
    occupation_basis_state,
    two_mode_toy_fermionic_hamiltonian,
)


def test_compare_fermionic_hamiltonian_reports_isospectral_mappings():
    report = compare_fermionic_hamiltonian(
        two_mode_toy_fermionic_hamiltonian(),
        label="two-mode",
        num_qubits=2,
        mode="exact",
        time_limit=10,
        step_counts=(1, 2),
        actual_errors=True,
        initial_state_kind="plus",
        actual_max_num_qubits=4,
        max_num_qubits=4,
    )

    assert report.is_isospectral
    assert report.spectrum.max_abs_difference < 1.0e-10
    np.testing.assert_allclose(
        report.spectrum.jw_eigenvalues,
        report.spectrum.bk_eigenvalues,
        atol=1.0e-10,
        rtol=1.0e-8,
    )

    assert report.jw_structure.num_qubits == 2
    assert report.bk_structure.num_qubits == 2
    assert report.jw_structure.num_non_identity_terms > 0
    assert report.bk_structure.num_non_identity_terms > 0
    assert (
        report.jw_structure.commuting_pairs
        + report.jw_structure.anticommute_pairs
        == report.jw_structure.total_pairs
    )
    assert report.jw_structure.qubit_wise_commuting_pairs <= report.jw_structure.commuting_pairs

    assert set(report.jw_trotter.first_order_bound_at_steps) == {1, 2}
    assert report.jw_actual_errors.first_order_state_infidelity is not None
    assert report.jw_actual_errors.first_order_frobenius_error[1] >= (
        report.jw_actual_errors.first_order_operator_error[1]
    )


def test_random_ordering_is_seeded():
    kwargs = dict(
        label="two-mode",
        num_qubits=2,
        mode="exact",
        time_limit=10,
        ordering_method="random",
        random_seed=17,
        step_counts=(1,),
        max_num_qubits=4,
    )

    left = compare_fermionic_hamiltonian(two_mode_toy_fermionic_hamiltonian(), **kwargs)
    right = compare_fermionic_hamiltonian(two_mode_toy_fermionic_hamiltonian(), **kwargs)

    assert left.jw_trotter.random_seed == 17
    assert right.jw_trotter.random_seed == 17
    assert left.jw_trotter.c1 == right.jw_trotter.c1
    assert left.jw_trotter.c2 == right.jw_trotter.c2
    assert left.bk_trotter.c1 == right.bk_trotter.c1
    assert left.bk_trotter.c2 == right.bk_trotter.c2


def test_occupation_basis_state_builds_one_hot_vectors():
    jw_state = occupation_basis_state(4, (0, 2), "JW")
    bk_state = occupation_basis_state(4, (0, 2), "BK")

    assert jw_state.shape == (16,)
    assert bk_state.shape == (16,)
    assert np.isclose(np.linalg.norm(jw_state), 1.0)
    assert np.isclose(np.linalg.norm(bk_state), 1.0)
    assert np.count_nonzero(jw_state) == 1
    assert np.count_nonzero(bk_state) == 1
