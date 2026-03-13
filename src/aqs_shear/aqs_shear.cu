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
Athermal Quasistatic Shear (AQS) implementation v2.1
- Fixed: Box and atom coordinate transformation consistency
- Box deformation now matches atom deformation correctly
------------------------------------------------------------------------------*/

#include "aqs_shear.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "minimize/minimizer_fire_box_change.cuh"
#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>  // For parsing comma-separated element names

// GPU kernel for applying shear deformation to atom positions
// Fixed: Use row-major matrix multiplication consistent with box transformation
static __global__ void gpu_apply_shear(
  const int N,
  const double mu_00, const double mu_01, const double mu_02,
  const double mu_10, const double mu_11, const double mu_12,
  const double mu_20, const double mu_21, const double mu_22,
  double* g_x, double* g_y, double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double x_old = g_x[i];
    double y_old = g_y[i];
    double z_old = g_z[i];
    
    // Row-major matrix multiplication: r_new = mu * r_old
    // This matches the box transformation: h_new = mu * h_old
    g_x[i] = mu_00 * x_old + mu_01 * y_old + mu_02 * z_old;
    g_y[i] = mu_10 * x_old + mu_11 * y_old + mu_12 * z_old;
    g_z[i] = mu_20 * x_old + mu_21 * y_old + mu_22 * z_old;
  }
}

AQSShear::AQSShear()
{
  params_.strain_per_step = 0.0;
  params_.total_strain = 0.0;
  params_.total_steps = 0;
  params_.minimizer = MinimizerType::FIRE;
  params_.shear_mode = ShearMode::SIMPLE;
  params_.force_tolerance = 1.0e-10;
  params_.max_minimize_steps = 1000;
  params_.dump_frequency = 1;
  params_.output_prefix = "aqs";
  params_.stress_file = "aqs_stress.txt";
  params_.atom_stress_file = "aqs_atom_stress.txt";
  params_.initial_relax = true;
  params_.dump_atom_stress = false;
}

void AQSShear::parse(const char** param, int num_param)
{
  printf("\n---------------------------------------------------------------\n");
  printf("Athermal Quasistatic Shear (AQS) simulation v2.1\n");
  printf("(Fixed: Box-atom transformation consistency)\n");
  printf("---------------------------------------------------------------\n");

  if (num_param < 5) {
    PRINT_INPUT_ERROR("aqsShear requires at least 4 parameters: direction strain_per_step total_strain minimizer\n");
  }

  parse_direction(param[1]);

  if (!is_valid_real(param[2], &params_.strain_per_step)) {
    PRINT_INPUT_ERROR("strain_per_step should be a number.\n");
  }
  if (params_.strain_per_step <= 0) {
    PRINT_INPUT_ERROR("strain_per_step should be positive.\n");
  }

  if (!is_valid_real(param[3], &params_.total_strain)) {
    PRINT_INPUT_ERROR("total_strain should be a number.\n");
  }
  if (params_.total_strain <= 0) {
    PRINT_INPUT_ERROR("total_strain should be positive.\n");
  }

  params_.total_steps = static_cast<int>(std::ceil(params_.total_strain / params_.strain_per_step));

  parse_minimizer(param[4]);
  parse_optional_params(param, num_param, 5);

  printf("    Direction: %s\n", params_.direction.c_str());
  printf("    Shear mode: %s\n", 
    params_.shear_mode == ShearMode::PURE ? "pure (symmetric)" : "simple (LAMMPS style)");
  printf("    Strain per step: %g (%.4f%%)\n", params_.strain_per_step, params_.strain_per_step * 100);
  printf("    Total strain: %g (%.2f%%)\n", params_.total_strain, params_.total_strain * 100);
  printf("    Total steps: %d\n", params_.total_steps);
  printf("    Minimizer: %s\n", 
    params_.minimizer == MinimizerType::SD ? "SD" : "FIRE");
  printf("    Force tolerance: %g eV/A\n", params_.force_tolerance);
  printf("    Max minimize steps: %d\n", params_.max_minimize_steps);
  printf("    Dump frequency: every %d step(s)\n", params_.dump_frequency);
  printf("    Output prefix: %s\n", params_.output_prefix.c_str());
  printf("    Stress file: %s\n", params_.stress_file.c_str());
  if (params_.dump_atom_stress) {
    printf("    Atom stress file: %s\n", params_.atom_stress_file.c_str());
  }
  printf("    Initial relax: %s\n", params_.initial_relax ? "yes" : "no");
  if (!params_.output_dir.empty()) {
    printf("    Output directory: %s\n", params_.output_dir.c_str());
  }
  if (!params_.element_names.empty()) {
    printf("    Element names:");
    for (const auto& elem : params_.element_names) {
      printf(" %s", elem.c_str());
    }
    printf("\n");
  }
  printf("---------------------------------------------------------------\n");
}

void AQSShear::parse_direction(const char* direction_str)
{
  params_.direction = direction_str;
  
  bool is_valid = false;
  const char* valid_directions[] = {"xy", "-xy", "xz", "-xz", "yz", "-yz"};
  for (const char* valid_dir : valid_directions) {
    if (strcmp(direction_str, valid_dir) == 0) {
      is_valid = true;
      break;
    }
  }
  
  if (!is_valid) {
    PRINT_INPUT_ERROR("Invalid direction. Use: xy, -xy, xz, -xz, yz, -yz\n");
  }
}

void AQSShear::parse_minimizer(const char* minimizer_str)
{
  if (strcmp(minimizer_str, "sd") == 0) {
    params_.minimizer = MinimizerType::SD;
  } else if (strcmp(minimizer_str, "fire") == 0) {
    params_.minimizer = MinimizerType::FIRE;
  } else {
    PRINT_INPUT_ERROR("Invalid minimizer. Use: sd or fire.\n");
  }
}

void AQSShear::parse_shear_mode(const char* mode_str)
{
  if (strcmp(mode_str, "pure") == 0) {
    params_.shear_mode = ShearMode::PURE;
  } else if (strcmp(mode_str, "simple") == 0) {
    params_.shear_mode = ShearMode::SIMPLE;
  } else {
    PRINT_INPUT_ERROR("Invalid shear mode. Use: pure or simple.\n");
  }
}

void AQSShear::parse_optional_params(const char** param, int num_param, int start_idx)
{
  int i = start_idx;
  while (i < num_param) {
    if (strcmp(param[i], "ftol") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("ftol requires a value.\n");
      if (!is_valid_real(param[i + 1], &params_.force_tolerance)) {
        PRINT_INPUT_ERROR("ftol should be a number.\n");
      }
      i += 2;
    } else if (strcmp(param[i], "max_steps") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("max_steps requires a value.\n");
      if (!is_valid_int(param[i + 1], &params_.max_minimize_steps)) {
        PRINT_INPUT_ERROR("max_steps should be an integer.\n");
      }
      i += 2;
    } else if (strcmp(param[i], "dump_every") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("dump_every requires a value.\n");
      if (!is_valid_int(param[i + 1], &params_.dump_frequency)) {
        PRINT_INPUT_ERROR("dump_every should be an integer.\n");
      }
      i += 2;
    } else if (strcmp(param[i], "prefix") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("prefix requires a value.\n");
      params_.output_prefix = param[i + 1];
      i += 2;
    } else if (strcmp(param[i], "stress") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("stress requires a value.\n");
      params_.stress_file = param[i + 1];
      i += 2;
    } else if (strcmp(param[i], "mode") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("mode requires pure or simple.\n");
      parse_shear_mode(param[i + 1]);
      i += 2;
    } else if (strcmp(param[i], "dump_atom_stress") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("dump_atom_stress requires true/false.\n");
      params_.dump_atom_stress = (strcmp(param[i + 1], "true") == 0);
      i += 2;
    } else if (strcmp(param[i], "initial_relax") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("initial_relax requires true/false.\n");
      params_.initial_relax = (strcmp(param[i + 1], "true") == 0);
      i += 2;
    } else if (strcmp(param[i], "elements") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("elements requires a comma-separated list (e.g., Cu,Zr).\n");
      // Parse comma-separated element names
      std::string elem_str = param[i + 1];
      std::stringstream ss(elem_str);
      std::string elem;
      params_.element_names.clear();
      while (std::getline(ss, elem, ',')) {
        params_.element_names.push_back(elem);
      }
      i += 2;
    } else if (strcmp(param[i], "out_dir") == 0) {
      if (i + 1 >= num_param) PRINT_INPUT_ERROR("out_dir requires a directory path.\n");
      params_.output_dir = param[i + 1];
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown optional parameter.\n");
    }
  }
}

double AQSShear::get_strain_sign()
{
  if (params_.direction[0] == '-') {
    return -1.0;
  }
  return 1.0;
}

int AQSShear::get_deform_components(int& i, int& j)
{
  std::string dir = params_.direction;
  if (dir[0] == '-') dir = dir.substr(1);
  
  if (dir == "xy") {
    i = 0; j = 1;
  } else if (dir == "xz") {
    i = 0; j = 2;
  } else if (dir == "yz") {
    i = 1; j = 2;
  } else {
    i = 0; j = 1;
  }
  return 0;
}

// FIXED: apply_shear now ensures consistent box and atom transformation
void AQSShear::apply_shear(Box& box, GPU_Vector<double>& position_per_atom, double strain_increment)
{
  int i, j;
  get_deform_components(i, j);
  double sign = get_strain_sign();
  strain_increment *= sign;

  // Build deformation gradient matrix F (identity + shear)
  // Note: We use row-major representation consistent with GPUMD's box matrix
  double F[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  
  if (params_.shear_mode == ShearMode::PURE) {
    // Pure shear: symmetric deformation
    F[i][j] = strain_increment;
    F[j][i] = strain_increment;
  } else {
    // Simple shear: LAMMPS style
    // For xy shear: x' = x + gamma * y, y' = y, z' = z
    // This means F[0][1] = gamma, keeping F diagonal as 1
    F[i][j] = strain_increment;
  }

  // Save old box
  double h_old[9];
  for (int idx = 0; idx < 9; ++idx) {
    h_old[idx] = box.cpu_h[idx];
  }

  // Update box: h_new = F * h_old (matrix multiplication)
  double h_new[9] = {0.0};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) {
        h_new[r * 3 + c] += F[r][k] * h_old[k * 3 + c];
      }
    }
  }
  
  for (int idx = 0; idx < 9; ++idx) {
    box.cpu_h[idx] = h_new[idx];
  }
  box.get_inverse();

  // Apply SAME transformation to atom positions
  const int N = position_per_atom.size() / 3;
  gpu_apply_shear<<<(N - 1) / 128 + 1, 128>>>(
    N,
    F[0][0], F[0][1], F[0][2],
    F[1][0], F[1][1], F[1][2],
    F[2][0], F[2][1], F[2][2],
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + 2 * N
  );
  GPU_CHECK_KERNEL
  
  printf("    Applied %s shear: strain increment = %g\n",
    params_.shear_mode == ShearMode::PURE ? "pure" : "simple", strain_increment);
}

void AQSShear::minimize_energy(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  
  if (params_.minimizer == MinimizerType::SD) {
    Minimizer_SD minimizer(-1, number_of_atoms, params_.max_minimize_steps, params_.force_tolerance);
    minimizer.compute(
      force, box, position_per_atom, type, group,
      potential_per_atom, force_per_atom, virial_per_atom);
  } else if (params_.minimizer == MinimizerType::FIRE) {
    // Use Box_Change version for shear deformation
    Minimizer_FIRE_Box_Change minimizer(number_of_atoms, params_.max_minimize_steps, params_.force_tolerance);
    minimizer.compute(
      force, box, position_per_atom, type, group,
      potential_per_atom, force_per_atom, virial_per_atom);
  }
}

void AQSShear::compute_stress_tensor(
  GPU_Vector<double>& virial_per_atom,
  Box& box,
  double stress[6])
{
  const int N = virial_per_atom.size() / 9;
  std::vector<double> virial_cpu(9 * N);
  virial_per_atom.copy_to_host(virial_cpu.data());
  
  double virial_sum[6] = {0.0};
  for (int n = 0; n < N; ++n) {
    virial_sum[0] += virial_cpu[n + 0 * N];  // sxx
    virial_sum[1] += virial_cpu[n + 1 * N];  // syy
    virial_sum[2] += virial_cpu[n + 2 * N];  // szz
    virial_sum[3] += virial_cpu[n + 3 * N];  // sxy
    virial_sum[4] += virial_cpu[n + 4 * N];  // syz
    virial_sum[5] += virial_cpu[n + 5 * N];  // sxz
  }
  
  double volume = box.get_volume();
  double factor = 160.2176621 / volume;
  
  for (int i = 0; i < 6; ++i) {
    stress[i] = virial_sum[i] * factor;
  }
}

void AQSShear::dump_structure(
  const std::string& filename,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<std::string>& element_names)  // Add element names parameter
{
  const int N = type.size();
  std::vector<double> pos_cpu(3 * N);
  std::vector<int> type_cpu(N);
  position_per_atom.copy_to_host(pos_cpu.data());
  type.copy_to_host(type_cpu.data());
  
  FILE* fid = fopen(filename.c_str(), "w");
  if (fid == NULL) {
    printf("Failed to open %s for writing.\n", filename.c_str());
    return;
  }
  
  fprintf(fid, "%d\n", N);
  // Fixed: Use column-major order consistent with GPUMD's dump_exyz format
  fprintf(fid, "pbc=\"T T T\" Lattice=\"%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\"\n",
    box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
    box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
    box.cpu_h[2], box.cpu_h[5], box.cpu_h[8]);
  
  // Use element symbols if available, otherwise use type numbers
  for (int n = 0; n < N; ++n) {
    int atom_type = type_cpu[n];
    if (atom_type >= 0 && atom_type < (int)element_names.size()) {
      fprintf(fid, "%s %.10f %.10f %.10f\n",
        element_names[atom_type].c_str(),
        pos_cpu[n],
        pos_cpu[n + N],
        pos_cpu[n + 2 * N]);
    } else {
      fprintf(fid, "%d %.10f %.10f %.10f\n",
        atom_type,
        pos_cpu[n],
        pos_cpu[n + N],
        pos_cpu[n + 2 * N]);
    }
  }
  
  fclose(fid);
}

void AQSShear::dump_atom_stress(
  const std::string& filename,
  int step,
  double strain,
  GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<int>& type)
{
  const int N = type.size();
  std::vector<double> virial_cpu(9 * N);
  std::vector<int> type_cpu(N);
  virial_per_atom.copy_to_host(virial_cpu.data());
  type.copy_to_host(type_cpu.data());
  
  double volume = box.get_volume();
  double factor = 160.2176621 / volume;
  
  char fname[256];
  snprintf(fname, 256, "%s_step_%04d.txt", filename.c_str(), step);
  
  FILE* fid = fopen(fname, "w");
  if (fid == NULL) return;
  
  fprintf(fid, "# step %d strain %.8f\n", step, strain);
  fprintf(fid, "# atom_id type sxx syy szz sxy syz sxz (GPa)\n");
  
  for (int n = 0; n < N; ++n) {
    fprintf(fid, "%d %d %.10f %.10f %.10f %.10f %.10f %.10f\n",
      n, type_cpu[n],
      virial_cpu[n + 0 * N] * factor,
      virial_cpu[n + 1 * N] * factor,
      virial_cpu[n + 2 * N] * factor,
      virial_cpu[n + 3 * N] * factor,
      virial_cpu[n + 4 * N] * factor,
      virial_cpu[n + 5 * N] * factor);
  }
  
  fclose(fid);
}

void AQSShear::dump_stress_header()
{
  FILE* fid = fopen(params_.stress_file.c_str(), "w");
  if (fid == NULL) {
    printf("Failed to open %s for writing.\n", params_.stress_file.c_str());
    return;
  }
  fprintf(fid, "# step strain sxx(GPa) syy(GPa) szz(GPa) sxy(GPa) syz(GPa) sxz(GPa)\n");
  fclose(fid);
}

void AQSShear::dump_stress_data(int step, double strain, double stress[6])
{
  FILE* fid = fopen(params_.stress_file.c_str(), "a");
  if (fid == NULL) return;
  fprintf(fid, "%d %.8f %.10f %.10f %.10f %.10f %.10f %.10f\n", 
    step, strain, stress[0], stress[1], stress[2], stress[3], stress[4], stress[5]);
  fclose(fid);
}

void AQSShear::compute(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  double stress[6] = {0.0};
  
  dump_stress_header();
  
  if (params_.initial_relax) {
    printf("\nInitial relaxation before AQS...\n");
    minimize_energy(force, box, position_per_atom, type, group,
                   potential_per_atom, force_per_atom, virial_per_atom);
    compute_stress_tensor(virial_per_atom, box, stress);
    printf("Initial relaxation completed.\n");
  }
  
  std::string init_file = params_.output_prefix + "_step_0000.xyz";
  if (!params_.output_dir.empty()) {
    init_file = params_.output_dir + "/" + init_file;
  }
  dump_structure(init_file, box, position_per_atom, type, params_.element_names);
  dump_stress_data(0, 0.0, stress);
  if (params_.dump_atom_stress) {
    dump_atom_stress(params_.atom_stress_file, 0, 0.0, virial_per_atom, box, type);
  }
  
  printf("\nStarting AQS shear simulation...\n");
  for (int step = 1; step <= params_.total_steps; ++step) {
    double current_strain = step * params_.strain_per_step;
    
    printf("\nStep %d/%d (strain = %.6f = %.4f%%)\n", 
           step, params_.total_steps, current_strain, current_strain * 100);
    
    printf("  Applying shear strain...\n");
    apply_shear(box, position_per_atom, params_.strain_per_step);
    
    printf("  Minimizing energy (%s)...\n",
      params_.minimizer == MinimizerType::SD ? "SD" : "FIRE");
    minimize_energy(force, box, position_per_atom, type, group,
                   potential_per_atom, force_per_atom, virial_per_atom);
    
    printf("  Computing stress...\n");
    compute_stress_tensor(virial_per_atom, box, stress);
    dump_stress_data(step, current_strain, stress);
    
    printf("  Stress: sxx=%.6f syy=%.6f szz=%.6f sxy=%.6f syz=%.6f sxz=%.6f GPa\n",
           stress[0], stress[1], stress[2], stress[3], stress[4], stress[5]);
    
    if (params_.dump_atom_stress && (step % params_.dump_frequency == 0)) {
      dump_atom_stress(params_.atom_stress_file, step, current_strain, virial_per_atom, box, type);
    }
    
    if (step % params_.dump_frequency == 0) {
      char filename[256];
      if (!params_.output_dir.empty()) {
        snprintf(filename, 256, "%s/%s_step_%04d.xyz", 
                 params_.output_dir.c_str(), params_.output_prefix.c_str(), step);
      } else {
        snprintf(filename, 256, "%s_step_%04d.xyz", 
                 params_.output_prefix.c_str(), step);
      }
      dump_structure(filename, box, position_per_atom, type, params_.element_names);
      printf("  Structure saved to %s\n", filename);
    }
  }
  
  printf("\n---------------------------------------------------------------\n");
  printf("AQS shear simulation completed!\n");
  printf("    Total strain applied: %.4f%%\n", params_.total_strain * 100);
  printf("    Final stress: sxx=%.6f syy=%.6f szz=%.6f sxy=%.6f syz=%.6f sxz=%.6f GPa\n",
         stress[0], stress[1], stress[2], stress[3], stress[4], stress[5]);
  printf("    Output files: %s_step_*.xyz\n", params_.output_prefix.c_str());
  printf("    Stress data: %s\n", params_.stress_file.c_str());
  if (params_.dump_atom_stress) {
    printf("    Atom stress files: %s_step_*.txt\n", params_.atom_stress_file.c_str());
  }
  printf("---------------------------------------------------------------\n");
}
