"""Tests for Pauli-string ordering behavior."""

import pytest

from analysis.ordering import (
    available_ordering_methods,
    pauli_commute,
    reorder_paulis,
)
from analysis.config_types import UnitaryConfiguration


TOY_HAMILTONIAN = {
    "ZZ": 0.2,
    "XI": -1.5,
    "IY": 0.7,
    "IZ": -0.1,
    "XX": 2.0,
    "II": 0.05,
}


def assert_same_terms(ordered, original):
    assert set(ordered) == set(original)
    for pauli_string, coefficient in ordered.items():
        assert coefficient == original[pauli_string]


def test_none_ordering_returns_dict_and_preserves_order():
    ordered = reorder_paulis(TOY_HAMILTONIAN, None)

    assert isinstance(ordered, dict)
    assert list(ordered.items()) == list(TOY_HAMILTONIAN.items())


def test_empty_ordering_input_returns_empty_dict():
    ordered = reorder_paulis({}, None)

    assert isinstance(ordered, dict)
    assert ordered == {}


@pytest.mark.parametrize(
    ("method", "expected_keys"),
    [
        ("identity", ["ZZ", "XI", "IY", "IZ", "XX", "II"]),
        ("reverse", ["II", "XX", "IZ", "IY", "XI", "ZZ"]),
        ("magnitude", ["XX", "XI", "IY", "ZZ", "IZ", "II"]),
        ("magnitude_descending", ["XX", "XI", "IY", "ZZ", "IZ", "II"]),
        ("magnitude_ascending", ["II", "IZ", "ZZ", "IY", "XI", "XX"]),
        ("lexicographical", ["II", "IY", "IZ", "XI", "XX", "ZZ"]),
        ("pauli_weight_ascending", ["II", "IY", "IZ", "XI", "XX", "ZZ"]),
        ("pauli_weight_descending", ["XX", "ZZ", "IY", "IZ", "XI", "II"]),
        ("diagonal_first", ["II", "IZ", "ZZ", "IY", "XI", "XX"]),
        ("diagonal_last", ["IY", "XI", "XX", "II", "IZ", "ZZ"]),
    ],
)
def test_toy_hamiltonian_ordering_methods(method, expected_keys):
    ordered = reorder_paulis(TOY_HAMILTONIAN, method)

    assert list(ordered) == expected_keys
    assert_same_terms(ordered, TOY_HAMILTONIAN)


def test_seeded_random_ordering_is_reproducible_and_preserves_terms():
    first = reorder_paulis(TOY_HAMILTONIAN, "random_seeded", seed=7)
    second = reorder_paulis(TOY_HAMILTONIAN, "random_seeded", seed=7)

    assert list(first) == list(second)
    assert list(first) != list(TOY_HAMILTONIAN)
    assert_same_terms(first, TOY_HAMILTONIAN)


def test_seeded_random_ordering_requires_seed():
    with pytest.raises(ValueError, match="requires an explicit seed"):
        reorder_paulis(TOY_HAMILTONIAN, "random_seeded")


def test_commuting_blocks_makes_commuting_terms_contiguous():
    hamiltonian = {
        "XX": 1.0,
        "ZI": 2.0,
        "YY": 3.0,
        "IZ": 4.0,
    }

    ordered = reorder_paulis(hamiltonian, "commuting_blocks")

    assert list(ordered) == ["XX", "YY", "ZI", "IZ"]
    assert pauli_commute("XX", "YY")
    assert pauli_commute("ZI", "IZ")
    assert_same_terms(ordered, hamiltonian)


def test_group_evolve_xyz_handles_y_only_terms():
    pauli_strings = {"ZII": 3.0, "YII": 2.0, "XII": 1.0}

    ordered = reorder_paulis(pauli_strings, "group_evolve_xyz")

    assert isinstance(ordered, dict)
    assert list(ordered) == ["XII", "YII", "ZII"]


def test_group_evolve_xyz_handles_identity_only_terms():
    pauli_strings = {"III": 0.5, "YII": 1.0, "ZII": 2.0}

    ordered = reorder_paulis(pauli_strings, "group_evolve_xyz")

    assert list(ordered) == ["III", "YII", "ZII"]


def test_group_evolve_xyz_rejects_mixed_pauli_terms():
    pauli_strings = {"XYZ": 1.0}

    with pytest.raises(ValueError, match="group_evolve_xyz can only be used"):
        reorder_paulis(pauli_strings, "group_evolve_xyz")


def test_invalid_method_reports_available_methods():
    with pytest.raises(ValueError) as exc_info:
        reorder_paulis(TOY_HAMILTONIAN, "not_an_ordering")

    message = str(exc_info.value)
    assert "not_an_ordering" in message
    assert "Available methods" in message
    assert "pauli_weight_ascending" in message


def test_available_ordering_methods_lists_new_methods():
    methods = available_ordering_methods()

    assert "identity" in methods
    assert "magnitude_descending" in methods
    assert "pauli_weight_ascending" in methods
    assert "diagonal_first" in methods
    assert "commuting_blocks" in methods


def test_ramped_trotter_config_records_ordering_seed():
    config = UnitaryConfiguration()

    config.encode_ramped_trotter(
        energy_error=1.0e-3,
        ordering_method="random_seeded",
        ordering_seed=17,
    )
    table = config._generate_TOML_table()

    assert config.ordering_seed == 17
    assert table["ordering_seed"] == 17
