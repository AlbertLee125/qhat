# h2_config.py

# General configuration
general.print_verbose()
general.file_stub = "h2_R0p7414_sto3g"
general.file_format = "default"
general.logfile = "h2_hamgen.log"

# H2 geometry
# Coordinates are effectively in Angstroms through PySCF's default Cartesian convention.
R = 0.7414
hamiltonian.add_atom("H", 0.0, 0.0, 0.0)
hamiltonian.add_atom("H", R,   0.0, 0.0)

# Basis
hamiltonian.basis = "sto-3g"

# Active space: for H2/STO-3G, use the full 4 spin-orbital space:
# 2 occupied spin orbitals + 2 vacant spin orbitals.
hamiltonian.num_active_occupied = 2
hamiltonian.num_active_vacant = 2

# Fermion-to-qubit mapping
hamiltonian.f2q_mapping = "JW"
# or:
# hamiltonian.f2q_mapping = "BK"