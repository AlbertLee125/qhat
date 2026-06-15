# QHAT Repository File Tree

This is a manual map of the repository source layout. It focuses on code, tests,
configuration, examples, and checked-in data. Generated or local-only artifacts
such as caches, logs, package metadata, and run-summary TOML files are omitted.

## Root

```text
qhat/
|-- README.md                         # Project overview
|-- LICENSE.md                        # License and release information
|-- pyproject.toml                    # Python package metadata and dependencies
|-- .gitignore                        # Git ignore rules
|-- .gitlab-ci.yml                    # GitLab CI configuration
|-- .github/
|   `-- workflows/
|       `-- python-app.yml            # GitHub Actions Python workflow
|-- time_evolution.py                 # Standalone CLI for exact and Trotter time evolution
|-- xxz_cs2cocl4_10.json              # Example Pauli Hamiltonian data
|-- analysis/                         # Python resource-analysis package
|-- common/                           # Shared Python modules used across QHAT
|-- hamiltonian_generator/            # Molecular Hamiltonian generation tools
`-- julia_trotter/                    # Standalone Julia Trotter demonstration
```

## Python Package Layout

The Python package is installed as `qhat`, with package directories configured in
`pyproject.toml`:

```text
qhat.analysis              -> analysis/
qhat.common                -> common/
qhat.hamiltonian_generator -> hamiltonian_generator/
```

## analysis/

`analysis/` contains the main resource-analysis workflow: load a Hamiltonian,
encode it as a unitary, build an algorithm, estimate resources, and optionally
write matrix output.

```text
analysis/
|-- README.md                         # Analysis package manual
|-- __init__.py                       # Package marker
|-- driver.py                         # Main analysis entry point: load config, run workflow
|-- config.py                         # Example/default analysis configuration
|-- config_types.py                   # Analysis configuration and State classes
|-- configuration.py                  # Loads and executes Python config scripts
|-- hamiltonian.py                    # Hamiltonian loaders and Hamiltonian wrapper
|-- unitary.py                        # Converts Hamiltonians into unitary encodings
|-- algorithm.py                      # Builds QPE and time-evolution algorithms
|-- analysis.py                       # Resource estimation and matrix-output dispatch
|-- file_io.py                        # Saves unitary matrices in NumPy, HDF5, or text form
|-- ordering.py                       # Pauli-term ordering helpers
|-- compare_mappings.py               # JW/BK spectrum and Trotter-coefficient comparison CLI
|-- trotter_coefficients.py           # Reference Trotter error coefficient estimator
|-- trotter_coefficients_fast.py      # Optimized exact/Monte Carlo Trotter estimator
|-- calibrate_throughput.py           # Calibrates exact-estimator throughput settings
|-- examples/
|   |-- config2.py                    # Example analysis config
|   `-- Be-H_1.30_sto-6g_as-003-003.tensors.npz
|                                      # Example second-quantized tensor Hamiltonian
`-- tests/
    |-- __init__.py                   # Test package marker
    |-- conftest.py                   # Test logging configuration
    |-- test_exact_computation.py     # Exact Trotter coefficient tests
    |-- test_exact_performance.py     # Exact-estimator performance tests
    |-- test_hamlib_loader.py         # HamLib HDF5 loader tests
    |-- test_compare_mappings.py      # JW/BK spectrum and Trotter comparison tests
    |-- test_modes.py                 # Trotter-estimator mode-selection tests
    |-- test_pauli_hamiltonian.py     # Pauli Hamiltonian loader and wrapper tests
    |-- test_throughput_config.py     # Throughput configuration tests
    |-- test_time_evolution_algorithms.py
    |                                  # Time-evolution algorithm matrix tests
    |-- data_pauli.json               # Test Pauli Hamiltonian JSON
    `-- hamlib_h2_sto6g.hdf5          # Test HamLib H2 Hamiltonian
```

Important Python files:

```text
analysis/driver.py                    # Top-level analysis workflow
analysis/config_types.py              # User-facing and internal config objects
analysis/hamiltonian.py               # HDF5, NumPy, HamLib, JSON, txt/dat Hamiltonian loading
analysis/unitary.py                   # Ramped Trotter and double-factorization encodings
analysis/algorithm.py                 # QPE, time evolution, controlled time evolution
analysis/analysis.py                  # Resource estimation through pyLIQTR or Cirq
analysis/compare_mappings.py          # Compare JW/BK spectra, Pauli structure, Trotter errors
```

## common/

`common/` holds reusable bloqs, Hamiltonian representations, Pauli utilities,
boson encodings, and Trotter implementations.

```text
common/
|-- README.md                         # Common-module manual and status table
|-- __init__.py                       # Package marker
|-- pauli_string_evolution.py         # Bloq for single Pauli-string evolution
|-- commuting_pauli_string_evolution.py
|                                      # Bloq for exact evolution of commuting Pauli sums
|-- dense_pauli_exp.py                # Dense Pauli string/exponential bloq helpers
|-- trotter_flattened.py              # Recommended flattened ramped Trotter implementation
|-- trotter_original.py               # Legacy nested ramped Trotter implementation
|-- LCPSHamiltonian.py                # Linear Combination of Pauli Strings representation
|-- MixedFermionBosonOperator.py      # Mixed fermion-boson Hamiltonian representation
|-- bosons_binary.py                  # Binary boson-to-qubit encoding helpers
|-- QPE_Kitaev.py                     # Exploratory Kitaev phase-estimation bloq
|-- pauli_utils.py                    # Pauli validation, matrices, and analytical evolution
|-- dps_test.py                       # Development script for dense Pauli helpers
|-- trotter_test.py                   # Development script for legacy Trotter code
|-- Untitled.ipynb                    # Notebook scratch/work file
`-- tests/
    |-- __init__.py                   # Test package marker
    |-- conftest.py                   # Test configuration
    |-- test_commuting_pauli_string_evolution.py
    |                                  # Tests for commuting Pauli evolution
    |-- test_dense_pauli_exp.py       # Tests for dense Pauli helpers
    |-- test_pauli_string_evolution.py
    |                                  # Tests for single Pauli-string evolution
    `-- test_trotter_flattened.py     # Tests for flattened Trotter implementation
```

Important Python files:

```text
common/pauli_string_evolution.py      # Core Pauli-string time-evolution bloq
common/commuting_pauli_string_evolution.py
                                      # Exact commuting-term evolution bloq
common/trotter_flattened.py           # Main Trotter implementation used by analysis
common/trotter_original.py            # Preserved legacy implementation
common/LCPSHamiltonian.py             # Pauli-string Hamiltonian container
common/MixedFermionBosonOperator.py   # Mixed fermionic and bosonic operator container
common/bosons_binary.py               # Bosonic binary encoding
common/logging_utils.py               # Shared logging setup and custom verbose level
common/pauli_utils.py                 # Shared matrix and Pauli-string utilities
```

## hamiltonian_generator/

`hamiltonian_generator/` generates molecular Hamiltonians with pySCF and
OpenFermion, applies active-space reductions, maps fermions to qubits, and writes
Pauli-string output.

```text
hamiltonian_generator/
|-- README.md                         # Hamiltonian-generation manual
|-- __init__.py                       # Package marker
|-- config.py                         # Example/default generation config
|-- hamgen.py                         # Main molecular Hamiltonian generation script
|-- hamgen_types.py                   # Generator configuration and State classes
|-- build_config.py                   # Builds batches of molecule config files
|-- eigendecompose.py                 # Converts Pauli sums to matrices and computes spectra
|-- eigenparse.py                     # Helper script for parsing/sorting eigenvalue data
`-- pauli_test.py                     # Development Trotter/Pauli test script
```

Important Python files:

```text
hamiltonian_generator/hamgen.py       # Hartree-Fock, active space, JW/BK mapping, file output
hamiltonian_generator/hamgen_types.py # User config classes for molecule generation
hamiltonian_generator/build_config.py # Batch config generation for molecular sweeps
hamiltonian_generator/eigendecompose.py
                                      # Dense-matrix spectral analysis for Pauli-string files
hamiltonian_generator/eigenparse.py   # Eigenvalue parsing utilities
```

## julia_trotter/

`julia_trotter/` is a standalone Julia demonstration for Trotterization and error
analysis, with checked-in Li-Li Hamiltonian data.

```text
julia_trotter/
|-- README.md                         # Julia demo manual
|-- Project.toml                      # Julia dependencies
|-- trotter_expts.jl                  # Main Julia Trotter demonstration script
|-- src/
|   |-- abstracttrotter.jl            # Abstract Trotter interfaces/helpers
|   |-- densesimulators.jl            # Dense-matrix simulation helpers
|   |-- error_bounds.jl               # Trotter error-bound computations
|   |-- experiment_utils.jl           # Experiment setup/reporting utilities
|   |-- hamiltonian_utils.jl          # Hamiltonian parsing and manipulation utilities
|   |-- parser.jl                     # Pauli Hamiltonian parser
|   |-- pauli_tensor_strings.jl       # Pauli tensor-string representation
|   |-- quantum_utils.jl              # General quantum helper functions
|   |-- statevector_simulators.jl     # State-vector simulation helpers
|   `-- tensorsimulators.jl           # Tensor-based simulation helpers
`-- Li-Li_jw/
    |-- Li-Li_2.90_hgbs-5_as-002-002_jw.dat
    |-- Li-Li_2.90_hgbs-5_as-002-004_jw.dat
    |-- Li-Li_2.90_hgbs-5_as-004-004_jw.dat
    |-- Li-Li_2.90_hgbs-5_as-004-006_jw.dat
    |-- Li-Li_2.90_hgbs-5_as-004-008_jw.dat
    |-- Li-Li_2.90_hgbs-5_as-006-008_jw.dat
    `-- Li-Li_2.90_hgbs-5_as-006-010_jw.dat
```

## Data And Configuration Files

```text
xxz_cs2cocl4_10.json                  # Root-level example spin Hamiltonian
analysis/examples/*.npz               # Example second-quantized tensor Hamiltonians
analysis/tests/*.json                 # Test Pauli Hamiltonian data
analysis/tests/*.hdf5                 # Test HamLib Hamiltonian data
julia_trotter/Li-Li_jw/*.dat          # Julia demo Pauli Hamiltonians
```

## Main Execution Paths

```text
analysis workflow:
  analysis/driver.py
    -> analysis/configuration.py
    -> analysis/hamiltonian.py
    -> analysis/unitary.py
    -> analysis/algorithm.py
    -> analysis/analysis.py

Hamiltonian generation workflow:
  hamiltonian_generator/hamgen.py
    -> hamiltonian_generator/hamgen_types.py
    -> OpenFermion / pySCF
    -> Pauli-string data files

Standalone Python time-evolution workflow:
  time_evolution.py
    -> Pauli-file/config loading
    -> exact matrix evolution
    -> first- or second-order Trotter comparison

Standalone Julia workflow:
  julia_trotter/trotter_expts.jl
    -> julia_trotter/src/*.jl
    -> julia_trotter/Li-Li_jw/*.dat
```
