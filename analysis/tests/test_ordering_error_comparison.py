"""Tests for the H, excited-H, and H2 ordering-error example."""

import networkx as nx
import numpy as np

from analysis.examples.ordering_error_comparison import (
    ORDERING_METHODS,
    build_evolution_cases,
    noncommutation_graph,
    particle_sector_basis_indices,
    run_ordering_experiments,
)


def test_evolution_cases_use_physical_particle_sectors():
    cases = build_evolution_cases()

    assert [case.name for case in cases] == ["H ground", "H excited", "H2 ground"]
    assert cases[0].energy < cases[1].energy

    for case in cases:
        sector = set(
            particle_sector_basis_indices(case.num_qubits, case.particle_number)
        )
        support = set(np.flatnonzero(abs(case.state) > 1e-10))
        assert support <= sector
        assert np.isclose(np.linalg.norm(case.state), 1.0)


def test_requested_orderings_produce_finite_error_diagnostics():
    cases = build_evolution_cases()
    results = run_ordering_experiments(cases=cases)

    assert len(results) == len(cases) * len(ORDERING_METHODS)
    assert {result.ordering_method for result in results} == set(ORDERING_METHODS)

    for result in results:
        assert np.isfinite(result.operator_error)
        assert np.isfinite(result.state_error)
        assert np.isfinite(result.infidelity)
        assert result.operator_error >= 0.0
        assert result.state_error >= 0.0
        assert 0.0 <= result.infidelity <= 1.0


def test_ground_and_excited_h_share_operator_error_but_not_state_metric():
    results = run_ordering_experiments()
    by_key = {
        (result.case_name, result.ordering_method): result
        for result in results
    }

    state_metric_differs = False
    for ordering in ORDERING_METHODS:
        ground = by_key[("H ground", ordering)]
        excited = by_key[("H excited", ordering)]
        assert np.isclose(ground.operator_error, excited.operator_error)
        state_metric_differs |= not np.isclose(
            ground.infidelity,
            excited.infidelity,
            rtol=1e-5,
            atol=1e-12,
        )

    assert state_metric_differs


def test_greedy_order_is_contiguous_by_commuting_color():
    cases = build_evolution_cases()
    results = run_ordering_experiments(
        cases=cases,
        ordering_methods=("group_evolve_greedy",),
    )

    for case, result in zip(cases, results):
        items = list(case.terms.items())
        graph = noncommutation_graph(case.terms)
        coloring = nx.greedy_color(graph)
        color_by_pauli = {
            items[node][0]: color
            for node, color in coloring.items()
        }
        ordered_colors = [color_by_pauli[pauli] for pauli in result.ordered_paulis]

        for color in set(ordered_colors):
            positions = [
                index
                for index, ordered_color in enumerate(ordered_colors)
                if ordered_color == color
            ]
            assert positions == list(range(min(positions), max(positions) + 1))
