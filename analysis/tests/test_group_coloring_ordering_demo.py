"""Tests for the graph-coloring ordering demo."""

from analysis.examples.group_coloring_ordering_demo import (
    build_greedy_coloring_groups,
    group_is_pairwise_commuting,
    h2_sto3g_jw_terms,
    noncommuting_pairs,
)


def test_h2_demo_noncommutation_graph_counts():
    terms = h2_sto3g_jw_terms(include_identity=False)
    graph, coloring, groups = build_greedy_coloring_groups(terms)

    assert len(terms) == 14
    assert graph.number_of_nodes() == 14
    assert graph.number_of_edges() == 16
    assert len(noncommuting_pairs(terms)) == 16
    assert max(coloring.values()) + 1 == 2
    assert sorted(len(group) for group in groups) == [4, 10]


def test_h2_demo_color_groups_are_pairwise_commuting():
    terms = h2_sto3g_jw_terms(include_identity=False)
    _, _, groups = build_greedy_coloring_groups(terms)

    assert all(group_is_pairwise_commuting(group) for group in groups)
