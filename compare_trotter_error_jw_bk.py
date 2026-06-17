#!/usr/bin/env python3
"""
Compare Trotter errors between Jordan-Wigner and Bravyi-Kitaev mappings.

Uses the Diatomic Lithium (Li₂) molecule example from analysis/examples/.

Although JW and BK have identical eigenspectra (isomorphic), they produce different
Pauli decompositions → different Trotter error coefficients.

Usage:
    python compare_trotter_error_jw_bk.py
"""

import sys
import logging
import numpy as np
from openfermion import InteractionOperator, jordan_wigner, bravyi_kitaev, QubitOperator

import logging
from qhat.common.logging_utils import add_verbose_level, VERBOSE

add_verbose_level()
logging.basicConfig(level=VERBOSE, format='%(levelname)-7s | %(message)s')

# Add analysis to path
sys.path.insert(0, 'analysis')
from trotter_coefficients_fast import trotter_error_estimator_fast


def load_hamiltonian(filename):
    """
    Load Diatomic Lithium molecule Hamiltonian from NumPy .npz file.
    
    File format (from hamiltonian_generator):
        - constant: scalar constant term (optional)
        - one_body: one-body tensor
        - two_body: two-body tensor
    
    Returns:
        InteractionOperator
    """
    print(f'Loading Hamiltonian from "{filename}"...')
    data = np.load(filename)
    
    # Extract tensors
    constant = data.get("constant", np.array(0.0))[()]  # Extract scalar from 0D array
    one_body = data["one_body"]
    two_body = data["two_body"]
    
    print(f"  Constant term  : {constant}")
    print(f"  One-body shape : {one_body.shape}")
    print(f"  Two-body shape : {two_body.shape}")
    print(f"  Number of spin orbitals : {one_body.shape[0]}")
    
    return InteractionOperator(constant, one_body, two_body)


def qubit_operator_to_terms_list(qubit_operator):
    """Convert QubitOperator to list of individual QubitOperators."""
    terms = []
    for pauli_string, coeff in qubit_operator.terms.items():
        if abs(coeff) > 1e-12:
            terms.append(QubitOperator(pauli_string, coeff))
    return terms


def main():
    print("=" * 70)
    print("TROTTER ERROR COMPARISON: Jordan-Wigner vs Bravyi-Kitaev")
    print("=" * 70)
    print()
    
    # Load Diatomic Lithium molecule
    filename = "hamiltonian_generator/diatomic_lithium_as-004-006.tensors.npz"
    
    try:
        fermion_ham = load_hamiltonian(filename)
    except FileNotFoundError:
        print(f'ERROR: Could not find "{filename}"')
        print("Make sure you're running from the qhat root directory.")
        print()
        print("Expected file structure:")
        print("  qhat/")
        print("  ├── compare_trotter_error_jw_bk.py  ← This script")
        print("  └── analysis/")
        print("      └── examples/")
        print("          └── diatomic_lithium_2.00_sto-3g_as-004-006.tensors.npz")
        return
    
    n_qubits = fermion_ham.n_qubits
    
    print()
    print(f"System: Diatomic Lithium (Li₂)")
    print(f"  Basis set: STO-3G")
    print(f"  Bond length: 2.00 Angstrom")
    print(f"  Active space: 4 occupied + 6 vacant spin orbitals")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Hilbert space dimension: {2**n_qubits}")
    print()
    
    # Map to qubits
    print("Applying Jordan-Wigner transformation...")
    jw_ham = jordan_wigner(fermion_ham)
    jw_terms = qubit_operator_to_terms_list(jw_ham)
    print(f"  JW Hamiltonian: {len(jw_terms)} Pauli terms")
    
    print("Applying Bravyi-Kitaev transformation...")
    bk_ham = bravyi_kitaev(fermion_ham)
    bk_terms = qubit_operator_to_terms_list(bk_ham)
    print(f"  BK Hamiltonian: {len(bk_terms)} Pauli terms")
    print()
    
    # Compute Trotter error coefficients
    time_limit = 60  # 60 seconds for each computation
    
    print("Computing Trotter error coefficients...")
    print(f"  Time limit: {time_limit}s per mapping")
    print(f"  Using fast exact/Monte Carlo hybrid computation")
    print()
    
    print("Jordan-Wigner mapping:")
    print("-" * 70)
    jw_c1, jw_c2 = trotter_error_estimator_fast(
        jw_terms, 
        time_limit,
        mode='monte_carlo',
        auto_exact=True  # Auto-switch to exact if feasible
    )
    print()
    
    print("Bravyi-Kitaev mapping:")
    print("-" * 70)
    bk_c1, bk_c2 = trotter_error_estimator_fast(
        bk_terms,
        time_limit,
        mode='monte_carlo',
        auto_exact=True  # Auto-switch to exact if feasible
    )
    print()
    
    # Display comparison
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Coefficient':<25} {'Jordan-Wigner':<20} {'Bravyi-Kitaev':<20} {'Ratio (BK/JW)':<15}")
    print("-" * 70)
    print(f"{'C1 (1st order)':<25} {jw_c1:<20.6f} {bk_c1:<20.6f} {bk_c1/jw_c1:<15.4f}")
    print(f"{'C2 (2nd order)':<25} {jw_c2:<20.6f} {bk_c2:<20.6f} {bk_c2/jw_c2:<15.4f}")
    print("-" * 70)
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    print()
    
    ratio_c1 = bk_c1 / jw_c1
    ratio_c2 = bk_c2 / jw_c2
    
    # Check if coefficients are similar (within 10%)
    if abs(ratio_c1 - 1.0) < 0.1 and abs(ratio_c2 - 1.0) < 0.1:
        print("  ≈ Trotter errors are very similar (~10% difference)")
        print("    Both mappings will require approximately the same number of Trotter steps")
    elif bk_c1 < jw_c1 and bk_c2 < jw_c2:
        pct_1st = (1 - bk_c1/jw_c1) * 100
        pct_2nd = (1 - bk_c2/jw_c2) * 100
        print("  ✅ Bravyi-Kitaev has LOWER Trotter errors!")
        print(f"     → {pct_1st:.1f}% reduction in 1st-order error coefficient")
        print(f"     → {pct_2nd:.1f}% reduction in 2nd-order error coefficient")
        print()
        print("  PRACTICAL IMPACT:")
        print("    - BK will need fewer Trotter steps for the same accuracy")
        print("    - BK may be more efficient for time evolution algorithms")
    else:
        pct_1st = (1 - jw_c1/bk_c1) * 100
        pct_2nd = (1 - jw_c2/bk_c2) * 100
        print("  ✅ Jordan-Wigner has LOWER Trotter errors!")
        print(f"     → {pct_1st:.1f}% reduction in 1st-order error coefficient")
        print(f"     → {pct_2nd:.1f}% reduction in 2nd-order error coefficient")
        print()
        print("  PRACTICAL IMPACT:")
        print("    - JW will need fewer Trotter steps for the same accuracy")
        print("    - JW may be more efficient for time evolution algorithms")
    
    print()
    print("KEY INSIGHT:")
    print("  While JW and BK are mathematically equivalent (same eigenvalues),")
    print("  they produce different Pauli decompositions with different commutator")
    print("  structures → different Trotter error bounds → different practical efficiency.")
    print()
    print("NOTE:")
    print("  These error coefficients determine the number of Trotter steps needed:")
    print("    1st order: r ≥ t²C₁/(2ε)  where ε is target error")
    print("    2nd order: r ≥ (t³C₂/ε)^(1/2)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()