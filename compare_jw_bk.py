"""
Comparison of Trotter Error: Jordan-Wigner vs Bravyi-Kitaev

Research Question: How does the choice of fermion-to-qubit encoding affect
Trotterization error, even though the encodings are isomorphic?

Expected Result: Different Pauli decompositions → different commutation 
structure → different error coefficients → different circuit resources
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from openfermion import QubitOperator

# Your QHAT imports
from qhat.analysis.trotter_coefficients_fast import trotter_error_estimator_fast
from qhat.analysis.ordering import reorder_paulis
from qhat.analysis.unitary import Trotterization
from qhat.io_utils import get_molecule_qubit_hamiltonian

# ============================================================================
# STEP 1: MOLECULE SELECTION
# ============================================================================

def get_test_molecules() -> List[Dict]:
    """
    Define a set of molecules for testing.
    Start small, scale up.
    
    Returns:
        List of molecule specifications
    """
    molecules = [
        # Small molecules (quick testing)
        {
            'name': 'H2',
            'geometry': [('H', (0, 0, 0)), ('H', (0, 0, 0.74))],
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
            'description': 'Hydrogen molecule - 2 qubits (JW), 2 qubits (BK)'
        },
        
        # Medium molecules (main experiments)
        {
            'name': 'LiH',
            'geometry': [('Li', (0, 0, 0)), ('H', (0, 0, 1.5949))],
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
            'description': 'Lithium hydride - 6 qubits'
        },
        
        {
            'name': 'BeH2',
            'geometry': [
                ('Be', (0, 0, 0)), 
                ('H', (-1.334, 0, 0)), 
                ('H', (1.334, 0, 0))
            ],
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
            'description': 'Beryllium hydride - 8 qubits'
        },
        
        # Larger molecules (if computational resources allow)
        {
            'name': 'H2O',
            'geometry': [
                ('O', (0, 0, 0)),
                ('H', (0.757, 0.586, 0)),
                ('H', (-0.757, 0.586, 0))
            ],
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
            'description': 'Water - 10 qubits'
        },
    ]
    
    return molecules


# ============================================================================
# STEP 2: HAMILTONIAN GENERATION
# ============================================================================

def get_hamiltonians_both_encodings(molecule: Dict) -> Tuple[QubitOperator, QubitOperator]:
    """
    Generate Hamiltonian in both JW and BK encodings.
    
    Args:
        molecule: Dictionary with molecule specification
    
    Returns:
        (hamiltonian_jw, hamiltonian_bk)
    """
    print(f"\n{'='*60}")
    print(f"Molecule: {molecule['name']} ({molecule['description']})")
    print(f"{'='*60}")
    
    # Jordan-Wigner encoding
    print("\n[1/2] Generating Jordan-Wigner Hamiltonian...")
    hamiltonian_jw = get_molecule_qubit_hamiltonian(
        geometry=molecule['geometry'],
        basis=molecule['basis'],
        charge=molecule['charge'],
        spin=molecule['spin'],
        mapping='jordan_wigner'  # ← JW encoding
    )
    
    # Bravyi-Kitaev encoding
    print("[2/2] Generating Bravyi-Kitaev Hamiltonian...")
    hamiltonian_bk = get_molecule_qubit_hamiltonian(
        geometry=molecule['geometry'],
        basis=molecule['basis'],
        charge=molecule['charge'],
        spin=molecule['spin'],
        mapping='bravyi_kitaev'  # ← BK encoding
    )
    
    return hamiltonian_jw, hamiltonian_bk


def analyze_hamiltonian_structure(hamiltonian: QubitOperator, encoding_name: str):
    """
    Print statistics about the Hamiltonian structure.
    
    This helps understand why Trotter errors differ.
    """
    terms = list(hamiltonian.terms.items())
    
    print(f"\n--- {encoding_name} Hamiltonian Structure ---")
    print(f"Number of terms: {len(terms)}")
    
    # Coefficient statistics
    coeffs = [abs(coeff) for _, coeff in terms]
    print(f"Coefficient range: [{min(coeffs):.6f}, {max(coeffs):.6f}]")
    print(f"Mean |coefficient|: {np.mean(coeffs):.6f}")
    print(f"Std |coefficient|:  {np.std(coeffs):.6f}")
    
    # Weight distribution (number of non-identity Paulis)
    def get_weight(pauli_string):
        """Count non-identity Paulis"""
        return sum(1 for pauli, _ in pauli_string if pauli != 'I')
    
    weights = [get_weight(pauli_string) for pauli_string, _ in terms]
    print(f"\nPauli string weight distribution:")
    from collections import Counter
    weight_counts = Counter(weights)
    for weight in sorted(weight_counts.keys()):
        count = weight_counts[weight]
        percentage = 100 * count / len(terms)
        print(f"  Weight {weight}: {count} terms ({percentage:.1f}%)")
    
    # Sample a few terms
    print(f"\nSample terms (first 5):")
    for i, (pauli_string, coeff) in enumerate(terms[:5]):
        # Convert to dense string format for readability
        dense_string = pauli_tuple_to_dense_string(pauli_string, hamiltonian.n_qubits)
        print(f"  {dense_string}: {coeff:.6f}")


def pauli_tuple_to_dense_string(pauli_tuple, n_qubits: int) -> str:
    """
    Convert OpenFermion sparse format to dense string.
    
    Example: ((0, 'X'), (3, 'Z')) with n_qubits=5 → "XIIIZ"
    """
    result = ['I'] * n_qubits
    for qubit_idx, pauli_type in pauli_tuple:
        result[qubit_idx] = pauli_type
    return ''.join(result)


# ============================================================================
# STEP 3: ERROR COEFFICIENT COMPUTATION
# ============================================================================

def compute_error_coefficients(
    hamiltonian: QubitOperator,
    encoding_name: str,
    ordering_methods: List[str] = None
) -> Dict:
    """
    Compute Trotter error coefficients for different orderings.
    
    Args:
        hamiltonian: QubitOperator in dense string format
        encoding_name: "JW" or "BK" (for labeling)
        ordering_methods: List of ordering strategies to test
    
    Returns:
        Dictionary with results for each ordering
    """
    if ordering_methods is None:
        ordering_methods = [None, 'magnitude', 'lexicographical', 'group_evolve_xyz']
    
    print(f"\n{'='*60}")
    print(f"Computing Error Coefficients: {encoding_name}")
    print(f"{'='*60}")
    
    results = {}
    
    for ordering_method in ordering_methods:
        method_name = ordering_method if ordering_method else "original"
        print(f"\n--- Ordering: {method_name} ---")
        
        try:
            # Reorder terms
            ordered_hamiltonian = reorder_paulis(
                hamiltonian.terms,  # generator of (pauli_string, coeff)
                ordering_method=ordering_method
            )
            
            # Convert to format expected by trotter_error_estimator_fast
            # Need: List of (dense_string, coefficient)
            pauli_terms = [
                (pauli_tuple_to_dense_string(ps, hamiltonian.n_qubits), coeff)
                for ps, coeff in ordered_hamiltonian.items()
            ]
            
            # Compute error coefficients
            print("  Computing C1, C21, C22 coefficients...")
            c1, c21, c22 = trotter_error_estimator_fast(pauli_terms)
            
            # Compute C1/2 (used for first vs second order decision)
            c1_half = c1 / 2.0
            
            print(f"  C1     = {c1:.6e}")
            print(f"  C1/2   = {c1_half:.6e}")
            print(f"  C21    = {c21:.6e}")
            print(f"  C22    = {c22:.6e}")
            
            # Determine which order is better
            if c1_half < c21 + c22:
                recommended_order = "first"
                better_coeff = c1_half
            else:
                recommended_order = "second"
                better_coeff = c21 + c22
            
            print(f"  → Recommended: {recommended_order} order (coeff = {better_coeff:.6e})")
            
            # Store results
            results[method_name] = {
                'c1': c1,
                'c1_half': c1_half,
                'c21': c21,
                'c22': c22,
                'recommended_order': recommended_order,
                'better_coeff': better_coeff,
                'pauli_terms': pauli_terms  # Store for later use
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[method_name] = {'error': str(e)}
    
    return results


# ============================================================================
# STEP 4: COMPARISON & VISUALIZATION
# ============================================================================

def compare_encodings(
    results_jw: Dict,
    results_bk: Dict,
    molecule_name: str
):
    """
    Compare error coefficients between JW and BK encodings.
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON: JW vs BK for {molecule_name}")
    print(f"{'='*60}")
    
    # Get common ordering methods
    orderings = [k for k in results_jw.keys() if 'error' not in results_jw[k]]
    
    print("\n{:<20} {:<15} {:<15} {:<15}".format(
        "Ordering", "JW C1/2", "BK C1/2", "Ratio (BK/JW)"
    ))
    print("-" * 65)
    
    for ordering in orderings:
        if ordering in results_jw and ordering in results_bk:
            jw_c1_half = results_jw[ordering]['c1_half']
            bk_c1_half = results_bk[ordering]['c1_half']
            ratio = bk_c1_half / jw_c1_half if jw_c1_half > 0 else float('inf')
            
            print("{:<20} {:<15.6e} {:<15.6e} {:<15.3f}".format(
                ordering, jw_c1_half, bk_c1_half, ratio
            ))
    
    print("\n{:<20} {:<15} {:<15} {:<15}".format(
        "Ordering", "JW (C21+C22)", "BK (C21+C22)", "Ratio (BK/JW)"
    ))
    print("-" * 65)
    
    for ordering in orderings:
        if ordering in results_jw and ordering in results_bk:
            jw_c2 = results_jw[ordering]['c21'] + results_jw[ordering]['c22']
            bk_c2 = results_bk[ordering]['c21'] + results_bk[ordering]['c22']
            ratio = bk_c2 / jw_c2 if jw_c2 > 0 else float('inf')
            
            print("{:<20} {:<15.6e} {:<15.6e} {:<15.3f}".format(
                ordering, jw_c2, bk_c2, ratio
            ))


def plot_comparison(
    results_jw: Dict,
    results_bk: Dict,
    molecule_name: str,
    save_path: str = None
):
    """
    Create visualization comparing JW vs BK error coefficients.
    """
    import matplotlib.pyplot as plt
    
    # Extract data
    orderings = [k for k in results_jw.keys() if 'error' not in results_jw[k]]
    
    jw_c1_half = [results_jw[o]['c1_half'] for o in orderings]
    bk_c1_half = [results_bk[o]['c1_half'] for o in orderings]
    
    jw_c2 = [results_jw[o]['c21'] + results_jw[o]['c22'] for o in orderings]
    bk_c2 = [results_bk[o]['c21'] + results_bk[o]['c22'] for o in orderings]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: C1/2 comparison
    x = np.arange(len(orderings))
    width = 0.35
    
    axes[0].bar(x - width/2, jw_c1_half, width, label='JW', alpha=0.8)
    axes[0].bar(x + width/2, bk_c1_half, width, label='BK', alpha=0.8)
    axes[0].set_xlabel('Ordering Method')
    axes[0].set_ylabel('C1/2 (First-Order Coefficient)')
    axes[0].set_title(f'{molecule_name}: First-Order Error Coefficient')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(orderings, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: C21 + C22 comparison
    axes[1].bar(x - width/2, jw_c2, width, label='JW', alpha=0.8)
    axes[1].bar(x + width/2, bk_c2, width, label='BK', alpha=0.8)
    axes[1].set_xlabel('Ordering Method')
    axes[1].set_ylabel('C21 + C22 (Second-Order Coefficient)')
    axes[1].set_title(f'{molecule_name}: Second-Order Error Coefficient')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(orderings, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# STEP 5: RESOURCE ESTIMATION (OPTIONAL)
# ============================================================================

def estimate_resources(
    pauli_terms: List[Tuple[str, float]],
    encoding_name: str,
    time: float = 1.0,
    num_steps: int = 10,
    method: str = "second order"
):
    """
    Estimate T-gate resources for Trotterization.
    
    Args:
        pauli_terms: List of (dense_string, coefficient)
        encoding_name: "JW" or "BK"
        time: Evolution time
        num_steps: Number of Trotter steps
        method: "first order" or "second order"
    """
    print(f"\n--- Resource Estimation: {encoding_name} ({method}) ---")
    
    try:
        # Create Trotterization bloq
        bloq = Trotterization.from_method(
            pauli_terms=pauli_terms,
            method=method,
            time=time,
            num_steps=num_steps,
            hbar=1.0
        )
        
        # Get T-complexity
        from qualtran.cirq_interop.t_complexity_protocol import t_complexity
        tc = t_complexity(bloq)
        
        print(f"  T-gates: {tc.t}")
        print(f"  Rotations: {tc.rotations}")
        print(f"  Clifford gates: {tc.clifford}")
        
        return {
            't_gates': tc.t,
            'rotations': tc.rotations,
            'clifford': tc.clifford
        }
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


# ============================================================================
# STEP 6: MAIN EXECUTION
# ============================================================================

def run_comparison_study():
    """
    Main function to run the complete comparison study.
    """
    print("="*60)
    print("TROTTER ERROR COMPARISON: JORDAN-WIGNER vs BRAVYI-KITAEV")
    print("="*60)
    
    # Get molecules to test
    molecules = get_test_molecules()
    
    # Choose which molecules to run (start with just H2 for testing)
    test_molecules = [molecules[0]]  # Start with H2
    # test_molecules = molecules[:2]  # H2 and LiH
    # test_molecules = molecules  # All molecules
    
    # Storage for all results
    all_results = {}
    
    for molecule in test_molecules:
        mol_name = molecule['name']
        
        # Generate Hamiltonians
        hamiltonian_jw, hamiltonian_bk = get_hamiltonians_both_encodings(molecule)
        
        # Analyze structure
        analyze_hamiltonian_structure(hamiltonian_jw, "Jordan-Wigner")
        analyze_hamiltonian_structure(hamiltonian_bk, "Bravyi-Kitaev")
        
        # Compute error coefficients
        results_jw = compute_error_coefficients(hamiltonian_jw, "JW")
        results_bk = compute_error_coefficients(hamiltonian_bk, "BK")
        
        # Compare
        compare_encodings(results_jw, results_bk, mol_name)
        
        # Visualize
        plot_comparison(results_jw, results_bk, mol_name, 
                       save_path=f"comparison_{mol_name}.png")
        
        # Store results
        all_results[mol_name] = {
            'jw': results_jw,
            'bk': results_bk,
            'molecule': molecule
        }
        
        # Optional: Resource estimation for best ordering
        print("\n" + "="*60)
        print(f"Resource Estimation: {mol_name}")
        print("="*60)
        
        # Find best ordering for each encoding
        best_ordering_jw = min(results_jw.items(), 
                              key=lambda x: x[1].get('better_coeff', float('inf')))[0]
        best_ordering_bk = min(results_bk.items(),
                              key=lambda x: x[1].get('better_coeff', float('inf')))[0]
        
        print(f"\nBest ordering for JW: {best_ordering_jw}")
        estimate_resources(
            results_jw[best_ordering_jw]['pauli_terms'],
            encoding_name="JW",
            method=results_jw[best_ordering_jw]['recommended_order']
        )
        
        print(f"\nBest ordering for BK: {best_ordering_bk}")
        estimate_resources(
            results_bk[best_ordering_bk]['pauli_terms'],
            encoding_name="BK",
            method=results_bk[best_ordering_bk]['recommended_order']
        )
    
    # Summary across all molecules
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL MOLECULES")
    print("="*60)
    
    for mol_name, data in all_results.items():
        print(f"\n{mol_name}:")
        results_jw = data['jw']
        results_bk = data['bk']
        
        # Find best coefficients
        best_jw = min(r.get('better_coeff', float('inf')) 
                     for r in results_jw.values() if 'better_coeff' in r)
        best_bk = min(r.get('better_coeff', float('inf'))
                     for r in results_bk.values() if 'better_coeff' in r)
        
        ratio = best_bk / best_jw
        winner = "JW" if best_jw < best_bk else "BK"
        
        print(f"  Best JW coeff: {best_jw:.6e}")
        print(f"  Best BK coeff: {best_bk:.6e}")
        print(f"  Ratio (BK/JW): {ratio:.3f}")
        print(f"  Winner: {winner} ({abs(1-ratio)*100:.1f}% better)")
    
    return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_comparison_study()
    
    # Optionally save results
    import pickle
    with open('jw_vs_bk_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\n✓ Results saved to: jw_vs_bk_results.pkl")