# _________________________________________________________________________________________________
# General configuration

general.print_verbose()                         # Additional information printed out
general.file_stub = "diatomic_lithium"          # Base name that all filenames are built from
general.file_format = "default"                 # Use default Pauli string style (not HamLib style)

# _________________________________________________________________________________________________
# Describe the Hamiltonian

# Li₂ is a diatomic molecule (two lithium atoms)
# Li-Li bond length is approximately 2.67 Å (experimental), using 2.0 Å for computation
bond_length = 2.0

# Place lithium atoms along the x-axis
hamiltonian.add_atom("Li", 0.0, 0.0, 0.0)              # First lithium at origin
hamiltonian.add_atom("Li", bond_length, 0.0, 0.0)     # Second lithium along x-axis

hamiltonian.basis = "sto-3g"                    # Select the atomic basis functions

# Li₂ has 6 electrons total (3 from each Li)
# Active space: 4 occupied + 6 vacant = 10 qubits total
hamiltonian.num_active_occupied = 4             # Specify the active space
hamiltonian.num_active_vacant = 6
