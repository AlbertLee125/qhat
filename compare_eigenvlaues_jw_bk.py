#!/usr/bin/env python3
"""
Compare eigenvalues between Jordan-Wigner and Bravyi-Kitaev mappings.

Uses the Be-H molecule example from analysis/examples/.

Demonstrates that JW and BK are isomorphic: identical eigenspectra.

Usage:
    python compare_eigenvalues_jw_bk.py
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from openfermion import InteractionOperator, jordan_wigner, bravyi_kitaev


def load_beh_hamiltonian(filename):
    """
    Load Be-H molecule Hamiltonian from NumPy .npz file.
    
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
    
    # Load Be-H molecule
    filename = "analysis/examples/Be-H_1.30_sto-6g_as-003-003.tensors.npz"
    
    try:
        fermion_ham = load_beh_hamiltonian(filename)
    except FileNotFoundError:
        print(f'ERROR: Could not find "{filename}"')
        print("Make sure you're running from the qhat root directory.")
        print()
        print("Expected file structure:")
        print("  qhat/")
        print("  ├── compare_eigenvalues_jw_bk.py  ← This script")
        print("  └── analysis/")
        print("      └── examples/")
        print("          └── Be-H_1.30_sto-6g_as-003-003.tensors.npz")
        return
    
    n_qubits = fermion_ham.n_qubits
    k = min(5, 2**n_qubits)  # Don't request more eigenvalues than dimension
    
    print()
    print(f"System: Be-H molecule (Beryllium Hydride)")
    print(f"  Basis set: STO-6G")
    print(f"  Bond length: 1.30 Angstrom")
    print(f"  Active space: 3 occupied + 3 vacant spin orbitals")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Hilbert space dimension: {2**n_qubits}")
    print()
    
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
        print()
        print("PHYSICAL INTERPRETATION:")
        print(f"  Ground state energy: {jw_eigenvalues[0]:.10f} Hartree")
        print(f"  First excitation:    {jw_eigenvalues[1] - jw_eigenvalues[0]:.10f} Hartree")
        if k >= 3:
            print(f"  Second excitation:   {jw_eigenvalues[2] - jw_eigenvalues[0]:.10f} Hartree")
    else:
        print("❌ WARNING: Eigenvalues differ.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()