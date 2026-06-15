#!/usr/bin/env python3
"""
Compare eigenvalues between Jordan-Wigner and Bravyi-Kitaev mappings.

Demonstrates that JW and BK are isomorphic: identical eigenspectra.

Usage:
    python compare_eigenvalues_jw_bk.py
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from openfermion import InteractionOperator, jordan_wigner, bravyi_kitaev


def create_hubbard_hamiltonian(n_sites=3, t=1.0, U=4.0):
    """
    Create an n-site Hubbard model Hamiltonian.
    
    H = -t Σ_{<i,j>,σ} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}
    """
    n_qubits = 2 * n_sites  # 2 spin orbitals per site
    
    # One-body: nearest-neighbor hopping (linear chain)
    one_body = np.zeros((n_qubits, n_qubits))
    for i in range(n_sites - 1):
        # Spin up hopping
        one_body[2*i, 2*(i+1)] = -t
        one_body[2*(i+1), 2*i] = -t
        # Spin down hopping
        one_body[2*i+1, 2*(i+1)+1] = -t
        one_body[2*(i+1)+1, 2*i+1] = -t
    
    # Two-body: on-site repulsion
    two_body = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    for site in range(n_sites):
        up = 2 * site
        down = 2 * site + 1
        two_body[up, down, down, up] = U
    
    return InteractionOperator(0.0, one_body, two_body)


def pauli_to_sparse_matrix(qubit_op):
    """Convert QubitOperator to sparse matrix."""
    from scipy.sparse import csr_matrix, lil_matrix
    
    # Get number of qubits
    n_qubits = 0
    for term in qubit_op.terms:
        if term:
            n_qubits = max(n_qubits, max(idx for idx, _ in term) + 1)
    
    dim = 2 ** n_qubits
    H = lil_matrix((dim, dim), dtype=complex)
    
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_matrices = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    for term, coeff in qubit_op.terms.items():
        # Build Pauli string matrix
        pauli_str = ['I'] * n_qubits
        for idx, op in term:
            pauli_str[idx] = op
        
        # Tensor product
        mat = pauli_matrices[pauli_str[0]]
        for op_char in pauli_str[1:]:
            mat = np.kron(mat, pauli_matrices[op_char])
        
        H += coeff * mat
    
    return H.tocsr()


def compute_eigenvalues(qubit_op, k=5):
    """Compute k smallest eigenvalues."""
    H_sparse = pauli_to_sparse_matrix(qubit_op)
    eigenvalues, _ = eigsh(H_sparse, k=k, which='SA')
    return sorted(eigenvalues)


def main():
    print("=" * 70)
    print("EIGENVALUE COMPARISON: Jordan-Wigner vs Bravyi-Kitaev")
    print("=" * 70)
    print()
    
    # Parameters
    n_sites = 3
    t = 1.0
    U = 4.0
    k = 5
    
    print(f"System: {n_sites}-site Hubbard model")
    print(f"  Parameters: t={t}, U={U}")
    print(f"  Qubits: {2*n_sites} (2 spin orbitals per site)")
    print()
    
    # Create Hamiltonian
    print("Creating fermionic Hamiltonian...")
    fermion_ham = create_hubbard_hamiltonian(n_sites, t, U)
    
    # Map to qubits
    print("Applying Jordan-Wigner transformation...")
    jw_ham = jordan_wigner(fermion_ham)
    print(f"  JW Hamiltonian: {len(jw_ham.terms)} Pauli terms")
    
    print("Applying Bravyi-Kitaev transformation...")
    bk_ham = bravyi_kitaev(fermion_ham)
    print(f"  BK Hamiltonian: {len(bk_ham.terms)} Pauli terms")
    print()
    
    # Compute eigenvalues
    print(f"Computing {k} lowest eigenvalues...")
    print()
    
    jw_eigenvalues = compute_eigenvalues(jw_ham, k=k)
    bk_eigenvalues = compute_eigenvalues(bk_ham, k=k)
    
    # Display results
    print("-" * 70)
    print(f"{'Index':<10} {'Jordan-Wigner':<25} {'Bravyi-Kitaev':<25} {'|Difference|':<12}")
    print("-" * 70)
    
    for i, (jw_eig, bk_eig) in enumerate(zip(jw_eigenvalues, bk_eigenvalues)):
        diff = abs(jw_eig - bk_eig)
        print(f"{i:<10} {jw_eig:<25.12f} {bk_eig:<25.12f} {diff:<12.2e}")
    
    print("-" * 70)
    print()
    
    # Check matching
    max_diff = max(abs(jw - bk) for jw, bk in zip(jw_eigenvalues, bk_eigenvalues))
    
    print(f"Maximum difference: {max_diff:.2e}")
    print()
    
    if max_diff < 1e-10:
        print("✅ SUCCESS: Eigenvalues match!")
        print("   JW and BK are isomorphic transformations.")
    else:
        print("❌ WARNING: Eigenvalues differ.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()