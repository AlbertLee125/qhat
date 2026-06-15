#!/usr/bin/env python3
"""
Compare Trotter errors between Jordan-Wigner and Bravyi-Kitaev mappings.

Although JW and BK have identical eigenspectra, they produce different
Pauli decompositions → different Trotter error coefficients.

Usage:
    python compare_trotter_error_jw_bk.py
"""

import sys
import numpy as np
from openfermion import InteractionOperator, jordan_wigner, bravyi_kitaev, QubitOperator

# Add analysis to path
sys.path.insert(0, 'analysis')
from trotter_coefficients_fast import trotter_error_estimator_fast


def create_hubbard_hamiltonian(n_sites=3, t=1.0, U=4.0):
    """
    Create an n-site Hubbard model Hamiltonian.
    
    H = -t Σ_{<i,j>,σ} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}
    """
    n_qubits = 2 * n_sites
    
    # One-body: nearest-neighbor hopping
    one_body = np.zeros((n_qubits, n_qubits))
    for i in range(n_sites - 1):
        # Spin up
        one_body[2*i, 2*(i+1)] = -t
        one_body[2*(i+1), 2*i] = -t
        # Spin down
        one_body[2*i+1, 2*(i+1)+1] = -t
        one_body[2*(i+1)+1, 2*i+1] = -t
    
    # Two-body: on-site repulsion
    two_body = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    for site in range(n_sites):
        up = 2 * site
        down = 2 * site + 1
        two_body[up, down, down, up] = U
    
    return InteractionOperator(0.0, one_body, two_body)


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
    
    # Parameters
    n_sites = 3
    t = 1.0
    U = 4.0
    time_limit = 30
    
    print(f"System: {n_sites}-site Hubbard model")
    print(f"  Parameters: t={t}, U={U}")
    print(f"  Qubits: {2*n_sites} (linear chain)")
    print()
    
    # Create Hamiltonian
    print("Creating fermionic Hamiltonian...")
    fermion_ham = create_hubbard_hamiltonian(n_sites, t, U)
    
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
    print("Computing Trotter error coefficients...")
    print(f"  Time limit: {time_limit}s per coefficient")
    print(f"  Using exact computation")
    print()
    
    print("Jordan-Wigner mapping:")
    print("-" * 40)
    jw_c1, jw_c2 = trotter_error_estimator_fast(
        jw_terms, 
        time_limit,
        mode='exact'
    )
    print()
    
    print("Bravyi-Kitaev mapping:")
    print("-" * 40)
    bk_c1, bk_c2 = trotter_error_estimator_fast(
        bk_terms,
        time_limit,
        mode='exact'
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
    
    if abs(bk_c1/jw_c1 - 1.0) < 0.01 and abs(bk_c2/jw_c2 - 1.0) < 0.01:
        print("  ≈ Trotter errors are very similar (~1% difference)")
    elif bk_c1 < jw_c1 and bk_c2 < jw_c2:
        pct_1st = (1 - bk_c1/jw_c1) * 100
        pct_2nd = (1 - bk_c2/jw_c2) * 100
        print("  ✅ Bravyi-Kitaev has LOWER Trotter errors!")
        print(f"     → {pct_1st:.1f}% reduction in 1st-order error")
        print(f"     → {pct_2nd:.1f}% reduction in 2nd-order error")
    else:
        pct_1st = (1 - jw_c1/bk_c1) * 100
        pct_2nd = (1 - jw_c2/bk_c2) * 100
        print("  ✅ Jordan-Wigner has LOWER Trotter errors!")
        print(f"     → {pct_1st:.1f}% reduction in 1st-order error")
        print(f"     → {pct_2nd:.1f}% reduction in 2nd-order error")
    
    print()
    print("KEY INSIGHT:")
    print("  While JW and BK have identical eigenvalues (isomorphic),")
    print("  they produce different Pauli decompositions with different")
    print("  commutator structures → different Trotter error bounds.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()