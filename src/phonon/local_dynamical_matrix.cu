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
Use finite difference to calculate the local (on-site) part of the dynamical
matrix with reduced storage/output: 3N rows × 3 columns.

For a full dynamical matrix, the elements are:
    D_ij^ab = -[F_i^a(+) - F_i^a(-)] / (2 * del * sqrt(m_i * m_j)) * conversion
where:
    - F_i^a(+) is the force on atom i in direction a when atom j is displaced
      by +del in direction b
    - F_i^a(-) is the force on atom i in direction a when atom j is displaced
      by -del in direction b
    - del is the displacement
    - m_i and m_j are the masses of atoms i and j
    - conversion is 1 for regular mode, or unit conversion factor for eskm mode

This file computes only the local (on-site) part, i.e. elements with i == j.
All off-diagonal elements (i != j) are discarded; only the on-site 3×3 block
for each atom is written. Output layout: 3N rows × 3 columns in local_dynmat.dat.

Atom order convention (must match util.py 1-to-1):
    - Atom index i = 0,1,...,N-1 corresponds to model.xyz row order (1st data row -> i=0, etc.).
    - Input: position_per_atom, type, mass use this order (GPUMD load from model.xyz).
    - Output: local_dynmat.dat rows (i,alpha) are written in the same order (i outer, alpha inner).
    - Downstream (e.g. util.py) assumes model.xyz row k <-> matrix block for atom k.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "local_dynamical_matrix.cuh"
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

void LocalDynamicalMatrix::displace_atom(
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

void LocalDynamicalMatrix::write_matrix_row(
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

void LocalDynamicalMatrix::convert_units_eskm()
{
  // ESKM mode: convert to 10 J/mol units
  // The same conversion as used in DynamicalMatrix.
  conversion = 9648.5;
}

void LocalDynamicalMatrix::parse(const char** param, size_t num_param)
{
  if (num_param != 3) {
    PRINT_INPUT_ERROR("compute_local_dynamical_matrix should have 2 parameters.\n");
  }

  // style: regular or eskm
  if (strcmp(param[1], "regular") == 0) {
    style = LOCAL_REGULAR;
    conversion = 1.0;
  } else if (strcmp(param[1], "eskm") == 0) {
    style = LOCAL_ESKM;
    convert_units_eskm();
  } else {
    PRINT_INPUT_ERROR("compute_local_dynamical_matrix style should be 'regular' or 'eskm'.\n");
  }

  // displacement
  if (!is_valid_real(param[2], &displacement)) {
    PRINT_INPUT_ERROR("displacement for compute_local_dynamical_matrix should be a number.\n");
  }
  if (displacement <= 0) {
    PRINT_INPUT_ERROR("displacement for compute_local_dynamical_matrix should be positive.\n");
  }

  printf(
    "Local dynamical matrix style: %s\n", style == LOCAL_REGULAR ? "regular" : "eskm");
  printf("Displacement for compute_local_dynamical_matrix = %g A.\n", displacement);
}

void LocalDynamicalMatrix::compute(
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

void LocalDynamicalMatrix::calculate_matrix(
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
  const int dynlen = 3; // only 3 columns (beta = 0,1,2) per row

  // Copy mass to CPU
  std::vector<double> cpu_mass(N);
  CHECK(gpuMemcpy(cpu_mass.data(), mass.data(), N * sizeof(double), gpuMemcpyDeviceToHost));

  // Open output file
  FILE* fp = fopen("local_dynmat.dat", "w");
  if (!fp) {
    PRINT_INPUT_ERROR("Cannot open local_dynmat.dat for writing.\n");
  }

  printf("Calculating Local Dynamical Matrix (on-site terms only)...\n");
  printf("  Total # of atoms = %d\n", N);
  printf("  Output size = %d x %d (3N rows x 3 cols, on-site only)\n", N * 3, dynlen);

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

      // Calculate local (on-site) dynamical matrix elements, i.e., only j == i.
      const double mass_i = cpu_mass[i];
      const double del2 = 2.0 * displacement;

      const int j = i;
      const double mass_j = cpu_mass[j];
      const double mass_factor = 1.0 / sqrt(mass_i * mass_j);

      for (int beta = 0; beta < 3; ++beta) {
        // Force storage: force[beta * N + j] for atom j, direction beta
        const int idx_force_pos = beta * N + j;
        const int idx_force_neg = beta * N + j;

        const double force_diff =
          force_negative[idx_force_neg] - force_positive[idx_force_pos];

        // Dynamical matrix element D_ii^ab
        const int dynmat_idx = beta; // only 3 columns
        dynmat_row[dynmat_idx] = force_diff / del2 * mass_factor * conversion;
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
  printf("Finished Calculating Local Dynamical Matrix\n");
  printf("Output written to local_dynmat.dat\n");
}

