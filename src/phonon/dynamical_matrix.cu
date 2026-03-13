/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Use finite difference to calculate the dynamical matrix.
    D_ij^ab = -[F_i^a(+) - F_i^a(-)] / (2 * del * sqrt(m_i * m_j)) * conversion
where:
    - F_i^a(+) is the force on atom i in direction a when atom j is displaced by +del in direction b
    - F_i^a(-) is the force on atom i in direction a when atom j is displaced by -del in direction b
    - del is the displacement
    - m_i and m_j are the masses of atoms i and j
    - conversion is 1 for regular mode, or unit conversion factor for eskm mode

Atom order convention (must match util.py 1-to-1):
    - Atom index i = 0,1,...,N-1 corresponds to model.xyz row order (1st data row -> i=0, etc.).
    - Input: position_per_atom, type, mass use this order (GPUMD load from model.xyz).
    - Output: dynmat.dat rows (i,alpha) are written in the same order (i outer, alpha inner).
    - Downstream (e.g. util.py) assumes model.xyz row k <-> matrix block for atom k.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "dynamical_matrix.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <vector>
#include <cstring>
#include <cmath>

static __global__ void gpu_shift_atom(const double dx, double* x) { x[0] += dx; }

void DynamicalMatrix::displace_atom(
  size_t atom_idx,
  int direction,
  double magnitude,
  GPU_Vector<double>& position_per_atom)
{
  const int N = position_per_atom.size() / 3;

  if (direction == 0) {
    gpu_shift_atom<<<1, 1>>>(magnitude, position_per_atom.data() + atom_idx);
    GPU_CHECK_KERNEL
  } else if (direction == 1) {
    gpu_shift_atom<<<1, 1>>>(magnitude, position_per_atom.data() + N + atom_idx);
    GPU_CHECK_KERNEL
  } else if (direction == 2) {
    gpu_shift_atom<<<1, 1>>>(magnitude, position_per_atom.data() + N * 2 + atom_idx);
    GPU_CHECK_KERNEL
  }
}

void DynamicalMatrix::write_matrix_row(
  FILE* fp,
  const std::vector<double>& dynmat_row)
{
  for (size_t j = 0; j < dynmat_row.size(); ++j) {
    if ((j + 1) % 3 == 0) {
      fprintf(fp, "%4.8f\n", dynmat_row[j]);
    } else {
      fprintf(fp, "%4.8f ", dynmat_row[j]);
    }
  }
}

void DynamicalMatrix::convert_units_eskm()
{
  // ESKM mode: convert to 10 J/mol units
  // In GPUMD, we use natural units where:
  // - Energy: eV
  // - Distance: Angstrom
  // - Mass: amu (atomic mass units)
  // 
  // Conversion to 10 J/mol:
  // 1 eV = 96.485 kJ/mol = 9648.5 * 10 J/mol
  // 1 Angstrom = 1 Angstrom
  // 1 amu = 1 g/mol
  // 
  // So: conversion = (9648.5 * 10 J/mol) / (1 Angstrom * 1 g/mol) = 9648.5
  // This matches LAMMPS metal units conversion
  conversion = 9648.5;
}

void DynamicalMatrix::parse(const char** param, size_t num_param)
{
  if (num_param != 3) {
    PRINT_INPUT_ERROR("compute_dynamical_matrix should have 2 parameters.\n");
  }

  // style: regular or eskm
  if (strcmp(param[1], "regular") == 0) {
    style = REGULAR;
    conversion = 1.0;
  } else if (strcmp(param[1], "eskm") == 0) {
    style = ESKM;
    convert_units_eskm();
  } else {
    PRINT_INPUT_ERROR("compute_dynamical_matrix style should be 'regular' or 'eskm'.\n");
  }

  // displacement
  if (!is_valid_real(param[2], &displacement)) {
    PRINT_INPUT_ERROR("displacement for compute_dynamical_matrix should be a number.\n");
  }
  if (displacement <= 0) {
    PRINT_INPUT_ERROR("displacement for compute_dynamical_matrix should be positive.\n");
  }

  printf("Dynamical matrix style: %s\n", style == REGULAR ? "regular" : "eskm");
  printf("Displacement for compute_dynamical_matrix = %g A.\n", displacement);
}

void DynamicalMatrix::compute(
  Force& force,
  Box& box,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& mass)
{
  number_of_atoms = type.size();
  calculate_matrix(
    force,
    box,
    cpu_position_per_atom,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom,
    mass);
}

void DynamicalMatrix::calculate_matrix(
  Force& force,
  Box& box,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& mass)
{
  const int N = number_of_atoms;
  const int dynlen = N * 3; // dynamical matrix dimension

  // Copy mass to CPU
  std::vector<double> cpu_mass(N);
  CHECK(gpuMemcpy(cpu_mass.data(), mass.data(), N * sizeof(double), gpuMemcpyDeviceToHost));

  // Open output file
  FILE* fp = fopen("dynmat.dat", "w");
  if (!fp) {
    PRINT_INPUT_ERROR("Cannot open dynmat.dat for writing.\n");
  }

  printf("Calculating Dynamical Matrix ...\n");
  printf("  Total # of atoms = %d\n", N);
  printf("  Total dynamical matrix elements = %d x %d\n", dynlen, dynlen);

  // Dynamical matrix storage: 3 rows (for alpha = 0, 1, 2) x dynlen columns
  std::vector<double> dynmat_row(dynlen, 0.0);

  // Force storage
  std::vector<double> force_positive(N * 3);
  std::vector<double> force_negative(N * 3);

  int progress = 0;
  const int progress_interval = (N > 10) ? (N / 10) : 1;

  // Loop over all atoms i
  for (int i = 0; i < N; ++i) {
    // Loop over directions alpha (x, y, z)
    for (int alpha = 0; alpha < 3; ++alpha) {
      // Clear the row
      std::fill(dynmat_row.begin(), dynmat_row.end(), 0.0);

      // Displace atom i in direction alpha by +del
      displace_atom(i, alpha, displacement, position_per_atom);

      // Compute forces
      force.compute(
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);

      // Copy forces to CPU
      CHECK(gpuMemcpy(
        force_positive.data(),
        force_per_atom.data(),
        N * 3 * sizeof(double),
        gpuMemcpyDeviceToHost));

      // Displace atom i in direction alpha by -2*del (from +del to -del)
      displace_atom(i, alpha, -2.0 * displacement, position_per_atom);

      // Compute forces
      force.compute(
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);

      // Copy forces to CPU
      CHECK(gpuMemcpy(
        force_negative.data(),
        force_per_atom.data(),
        N * 3 * sizeof(double),
        gpuMemcpyDeviceToHost));

      // Restore atom i to original position
      displace_atom(i, alpha, displacement, position_per_atom);

      // Calculate dynamical matrix elements
      // Following LAMMPS implementation:
      // 1. First displacement (+del): dynmat[alpha][j*3+beta] -= f[j][beta]  -> dynmat = -F_j^beta(+)
      // 2. Second displacement (-del): dynmat[alpha][j*3+beta] -= -f[j][beta]  -> dynmat = F_j^beta(-) - F_j^beta(+)
      // 3. Divide by (2*del*imass) and multiply by conversion
      // Final: D_ji^ab = [F_j^b(-) - F_j^b(+)] / (2*del*sqrt(m_i*m_j)) * conversion
      const double mass_i = cpu_mass[i];
      const double del2 = 2.0 * displacement;

      for (int j = 0; j < N; ++j) {
        const double mass_j = cpu_mass[j];
        const double mass_factor = 1.0 / sqrt(mass_i * mass_j);

        for (int beta = 0; beta < 3; ++beta) {
          // Force storage: force[beta * N + j] for atom j, direction beta
          const int idx_force_pos = beta * N + j;
          const int idx_force_neg = beta * N + j;

          // Following LAMMPS: first subtract F_j^beta(+), then add F_j^beta(-)
          // So: dynmat = -F_j^beta(+) + F_j^beta(-) = F_j^beta(-) - F_j^beta(+)
          const double force_diff =
            force_negative[idx_force_neg] - force_positive[idx_force_pos];

          // Dynamical matrix element D_ji^ab
          // Output format: for each row (i, alpha), we output columns (j, beta)
          const int dynmat_idx = j * 3 + beta;
          dynmat_row[dynmat_idx] = force_diff / del2 * mass_factor * conversion;
        }
      }

      // Write the row to file
      write_matrix_row(fp, dynmat_row);
    }

    // Progress update
    if ((i + 1) % progress_interval == 0) {
      progress = ((i + 1) * 100) / N;
      printf("  %d%% completed\n", progress);
      fflush(stdout);
    }
  }

  fclose(fp);
  printf("Finished Calculating Dynamical Matrix\n");
  printf("Output written to dynmat.dat\n");
}
