# NEB (Nudged Elastic Band) Implementation for GPUMD

## Overview

This implementation provides the Nudged Elastic Band (NEB) method for finding transition states and minimum energy paths between two configurations in molecular dynamics simulations.

## Why NEB in GPUMD?

The NEB implementation in GPUMD offers significant advantages when using NEP (Neuroevolution Potential) potentials:

1. **Speed**: NEP potentials on GPU are extremely fast, making NEB calculations efficient
2. **Accuracy**: NEP provides high-accuracy force calculations
3. **Integration**: Seamless integration with GPUMD's existing infrastructure

LAMMPS struggles with NEP potentials because:
- NEP requires specialized GPU kernels
- LAMMPS's NEB is designed for traditional force fields
- Communication overhead between LAMMPS and external NEP libraries

## Theory

The NEB method works by:

1. Creating a chain of replicas (images) between initial and final states
2. For each interior replica, computing:
   - The tangent vector along the path (using improved tangent method)
   - The perpendicular component of the true force
   - The spring force along the tangent to maintain equal spacing
3. Minimizing the band to find the minimum energy path
4. Optionally using climbing image NEB to find the exact saddle point

## Implementation Details

### Core Algorithm

The implementation follows the improved NEB method by Henkelman et al.:

- **Tangent calculation**: Uses the improved tangent method that handles energy extrema correctly
- **Force projection**: Removes the parallel component of the true force and adds spring force
- **Climbing image**: Inverts the parallel force component at the highest energy replica

### Key Features

1. **GPU Acceleration**: All computations use CUDA kernels for maximum performance
2. **Flexible Endpoints**: Support for free-end NEB
3. **Multiple Modes**: neighbor, ideal, and equal spacing modes
4. **Convergence Control**: Configurable energy and force tolerances

## Usage

### Basic Syntax

```
neb <nreplica> <etol> <ftol> <n1steps> <n2steps> <nevery> [keywords]
```

### Parameters

- `nreplica`: Number of replicas (images) along the path (minimum 3)
- `etol`: Energy tolerance for convergence (eV)
- `ftol`: Force tolerance for convergence (eV/Angstrom)
- `n1steps`: Maximum steps for regular NEB stage
- `n2steps`: Maximum steps for climbing NEB stage (0 to skip)
- `nevery`: Output frequency

### Keywords

- `spring <value>`: Spring constant (default 1.0 eV/Angstrom^2)
- `final <filename>`: Final configuration file (required)
- `free_end_ini [kspring]`: Free end for initial replica
- `free_end_final [kspring]`: Free end for final replica
- `mode <neighbor|ideal|equal>`: NEB mode

### Example

```
potential nep.txt
neb 7 1.0e-6 1.0e-4 1000 1000 10 final state_final.xyz spring 5.0
```

## Output

The NEB calculation produces:

1. **Console output**: Energy and force information for each replica at each output step
2. **neb_final_path.xyz**: Final path with all replica configurations

## Comparison with LAMMPS NEB

| Feature | GPUMD NEB | LAMMPS NEB |
|---------|-----------|------------|
| GPU Acceleration | Native | Limited |
| NEP Support | Native | Requires interface |
| Parallel Replicas | On single GPU | MPI-based |
| Memory Efficiency | High (shared GPU memory) | Lower |
| Ease of Use | Simple command | Complex setup |

## References

1. G. Henkelman, B. P. Uberuaga, and H. Jonsson, "A climbing image nudged elastic band method for finding saddle points and minimum energy paths," J. Chem. Phys. 113, 9901 (2000)

2. G. Henkelman and H. Jonsson, "Improved tangent estimate in the nudged elastic band method for finding minimum energy paths and saddle points," J. Chem. Phys. 113, 9978 (2000)

## Implementation Files

- `neb.cuh`: Header file with class definition
- `neb.cu`: Implementation with CUDA kernels
- `run.cu`: Modified to include NEB command parsing
- `makefile`: Updated to include NEB source files
