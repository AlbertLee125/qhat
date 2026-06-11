"""
Simple examples and tests for Jordan-Wigner and Bravyi-Kitaev spectral comparison.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
from openfermion import FermionOperator, InteractionOperator

from mapping_isomorphism import (
    compare_jordan_wigner_bravyi_kitaev_spectra,
    assert_jordan_wigner_bravyi_kitaev_isospectral,
)


# ============================================================================
# SIMPLE HAMILTONIAN GENERATORS
# ============================================================================

def create_constant_hamiltonian(constant=1.0, num_qubits=2):
    """
    Simplest possible Hamiltonian: just a constant (identity operator).
    H = constant * I
    """
    return FermionOperator((), constant)


def create_number_operator_hamiltonian(num_sites=2):
    """
    Simple number operator Hamiltonian.
    H = sum_i n_i where n_i = a^†_i a_i
    """
    hamiltonian = FermionOperator()
    for i in range(num_sites):
        hamiltonian += FermionOperator(f'{i}^ {i}', 1.0)
    return hamiltonian


def create_hopping_hamiltonian(num_sites=4, periodic=False):
    """
    Simple tight-binding (hopping) Hamiltonian.
    H = -t * sum_i (a^†_i a_{i+1} + a^†_{i+1} a_i)
    """
    t = 1.0
    hamiltonian = FermionOperator()
    
    for i in range(num_sites - 1):
        hamiltonian += FermionOperator(f'{i}^ {i+1}', -t)
        hamiltonian += FermionOperator(f'{i+1}^ {i}', -t)
    
    if periodic and num_sites > 2:
        hamiltonian += FermionOperator(f'{num_sites-1}^ 0', -t)
        hamiltonian += FermionOperator(f'0^ {num_sites-1}', -t)
    
    return hamiltonian


def create_hubbard_hamiltonian(num_sites=2, t=1.0, U=2.0):
    """
    Simple Fermi-Hubbard model.
    """
    hamiltonian = FermionOperator()
    
    for spin in range(2):
        for i in range(num_sites - 1):
            idx_i = 2 * i + spin
            idx_j = 2 * (i + 1) + spin
            hamiltonian += FermionOperator(f'{idx_i}^ {idx_j}', -t)
            hamiltonian += FermionOperator(f'{idx_j}^ {idx_i}', -t)
    
    for i in range(num_sites):
        idx_up = 2 * i
        idx_down = 2 * i + 1
        hamiltonian += FermionOperator(f'{idx_up}^ {idx_up} {idx_down}^ {idx_down}', U)
    
    return hamiltonian


def create_h2_minimal_hamiltonian():
    """
    Minimal H2 molecule Hamiltonian.
    """
    one_body_integrals = np.array([
        [-1.25, 0.0],
        [0.0, -1.25]
    ])
    
    two_body_integrals = np.zeros((2, 2, 2, 2))
    two_body_integrals[0, 0, 0, 0] = 0.5
    two_body_integrals[1, 1, 1, 1] = 0.5
    two_body_integrals[0, 1, 0, 1] = 0.25
    two_body_integrals[1, 0, 1, 0] = 0.25
    
    constant = 0.7
    
    return InteractionOperator(constant, one_body_integrals, two_body_integrals)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_constant_hamiltonian():
    """Test that a constant Hamiltonian gives matching spectra."""
    print("\n" + "="*70)
    print("TEST: Constant Hamiltonian")
    print("="*70)
    
    hamiltonian = create_constant_hamiltonian(constant=5.0, num_qubits=2)
    result = compare_jordan_wigner_bravyi_kitaev_spectra(hamiltonian, num_qubits=2)
    
    print(f"Number of qubits: {result.num_qubits}")
    print(f"Is isospectral: {result.is_isospectral}")
    print(f"Max difference: {result.max_abs_difference:.2e}")
    
    assert result.is_isospectral, "Constant Hamiltonian should be isospectral!"
    print("✓ Test passed!")
    return result


def test_number_operator():
    """Test number operator Hamiltonian."""
    print("\n" + "="*70)
    print("TEST: Number Operator Hamiltonian")
    print("="*70)
    
    hamiltonian = create_number_operator_hamiltonian(num_sites=3)
    result = assert_jordan_wigner_bravyi_kitaev_isospectral(hamiltonian)
    
    print(f"Number of qubits: {result.num_qubits}")
    print(f"Max difference: {result.max_abs_difference:.2e}")
    print("✓ Test passed!")
    return result


def test_hopping_hamiltonian():
    """Test tight-binding Hamiltonian."""
    print("\n" + "="*70)
    print("TEST: Hopping (Tight-Binding) Hamiltonian")
    print("="*70)
    
    hamiltonian = create_hopping_hamiltonian(num_sites=4, periodic=False)
    result = compare_jordan_wigner_bravyi_kitaev_spectra(hamiltonian)
    
    print(f"Number of qubits: {result.num_qubits}")
    print(f"Is isospectral: {result.is_isospectral}")
    print(f"Max difference: {result.max_abs_difference:.2e}")
    print(f"Ground state energy: {result.jw_eigenvalues[0]:.6f}")
    
    assert result.is_isospectral
    print("✓ Test passed!")
    return result


def test_hubbard_model():
    """Test Fermi-Hubbard model."""
    print("\n" + "="*70)
    print("TEST: Fermi-Hubbard Model")
    print("="*70)
    
    hamiltonian = create_hubbard_hamiltonian(num_sites=2, t=1.0, U=2.0)
    result = compare_jordan_wigner_bravyi_kitaev_spectra(hamiltonian)
    
    print(f"Number of qubits: {result.num_qubits}")
    print(f"Is isospectral: {result.is_isospectral}")
    print(f"Max difference: {result.max_abs_difference:.2e}")
    
    assert result.is_isospectral
    print("✓ Test passed!")
    return result


def test_h2_molecule():
    """Test H2 molecule Hamiltonian."""
    print("\n" + "="*70)
    print("TEST: H2 Molecule (InteractionOperator)")
    print("="*70)
    
    hamiltonian = create_h2_minimal_hamiltonian()
    result = compare_jordan_wigner_bravyi_kitaev_spectra(hamiltonian)
    
    print(f"Number of qubits: {result.num_qubits}")
    print(f"Is isospectral: {result.is_isospectral}")
    print(f"Max difference: {result.max_abs_difference:.2e}")
    
    assert result.is_isospectral
    print("✓ Test passed!")
    return result


def compare_different_system_sizes():
    """Compare spectra for different system sizes."""
    print("\n" + "="*70)
    print("COMPARISON: System Size Scaling")
    print("="*70)
    
    for num_sites in [2, 3, 4]:
        hamiltonian = create_number_operator_hamiltonian(num_sites)
        result = compare_jordan_wigner_bravyi_kitaev_spectra(hamiltonian)
        
        print(f"\nSystem size: {num_sites} sites ({result.num_qubits} qubits)")
        print(f"  Hilbert space dimension: {2**result.num_qubits}")
        print(f"  Is isospectral: {result.is_isospectral}")
        print(f"  Max eigenvalue difference: {result.max_abs_difference:.2e}")


def visualize_spectrum_comparison():
    """Create a visual comparison of JW and BK spectra."""
    print("\n" + "="*70)
    print("VISUALIZATION: Spectrum Comparison")
    print("="*70)
    
    hamiltonian = create_hopping_hamiltonian(num_sites=3)
    result = compare_jordan_wigner_bravyi_kitaev_spectra(hamiltonian)
    
    print(f"\nHamiltonian: 3-site hopping chain")
    print(f"Eigenvalue comparison (JW vs BK):")
    print("-" * 50)
    print(f"{'Index':<8} {'JW':<15} {'BK':<15} {'Difference':<12}")
    print("-" * 50)
    
    for i in range(min(10, len(result.jw_eigenvalues))):
        diff = result.jw_eigenvalues[i] - result.bk_eigenvalues[i]
        print(f"{i:<8} {result.jw_eigenvalues[i]:<15.8f} "
              f"{result.bk_eigenvalues[i]:<15.8f} {diff:<12.2e}")
    
    if len(result.jw_eigenvalues) > 10:
        print("...")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("JORDAN-WIGNER vs BRAVYI-KITAEV SPECTRAL COMPARISON TESTS")
    print("="*70)
    
    try:
        test_constant_hamiltonian()
        test_number_operator()
        test_hopping_hamiltonian()
        test_hubbard_model()
        test_h2_molecule()
        
        compare_different_system_sizes()
        visualize_spectrum_comparison()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nConclusion: Jordan-Wigner and Bravyi-Kitaev transformations")
        print("produce isospectral Hamiltonians (same eigenvalues) for all")
        print("tested fermionic systems, confirming they are valid isomorphisms.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()