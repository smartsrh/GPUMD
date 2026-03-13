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
Shear deformation implementation - FULL VERSION with C-style strings
Usage: shear <strain_rate> <direction> [dump_interval] [output_dir] [file_prefix]
------------------------------------------------------------------------------*/

#include "shear.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>

// GPU kernel for applying shear to atom positions
static __global__ void gpu_apply_shear_to_atoms(
  const int N,
  const int i,
  const int j,
  const double strain,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double r[3];
    r[0] = g_x[n];
    r[1] = g_y[n];
    r[2] = g_z[n];
    r[i] += strain * r[j];
    g_x[n] = r[0];
    g_y[n] = r[1];
    g_z[n] = r[2];
  }
}

Shear::Shear()
{
  active_ = false;
  strain_rate_ = 0.0;
  direction_[0] = '\0';
  shear_i_ = 0;
  shear_j_ = 1;
  strain_sign_ = 1.0;
  dump_interval_ = 1000;
  output_dir_[0] = '.';
  output_dir_[1] = '/';
  output_dir_[2] = '\0';
  file_prefix_[0] = 's';
  file_prefix_[1] = 'h';
  file_prefix_[2] = 'e';
  file_prefix_[3] = 'a';
  file_prefix_[4] = 'r';
  file_prefix_[5] = '\0';
  current_step_ = 0;
}

void Shear::parse_direction(const char* direction_str)
{
  // Copy direction string (max 3 chars: xy, -xy, etc.)
  strncpy(direction_, direction_str, sizeof(direction_) - 1);
  direction_[sizeof(direction_) - 1] = '\0';
  
  strain_sign_ = 1.0;
  const char* dir_ptr = direction_str;
  if (direction_str[0] == '-') {
    strain_sign_ = -1.0;
    dir_ptr = direction_str + 1;
  }
  
  if (strcmp(dir_ptr, "xy") == 0) {
    shear_i_ = 0;
    shear_j_ = 1;
  } else if (strcmp(dir_ptr, "xz") == 0) {
    shear_i_ = 0;
    shear_j_ = 2;
  } else if (strcmp(dir_ptr, "yz") == 0) {
    shear_i_ = 1;
    shear_j_ = 2;
  } else {
    PRINT_INPUT_ERROR("Invalid shear direction. Use: xy, -xy, xz, -xz, yz, -yz\n");
  }
}

void Shear::parse(const char** param, int num_param)
{
  printf("\n---------------------------------------------------------------\n");
  printf("Shear deformation with temperature control\n");
  printf("---------------------------------------------------------------\n");

  if (num_param < 3 || num_param > 6) {
    PRINT_INPUT_ERROR("shear requires 2-5 parameters:\n"
      "  shear <strain_rate> <direction> [dump_interval] [output_dir] [file_prefix]\n"
      "  Example: shear 0.00001 xy 500 ./output run1\n");
  }

  // Parse strain rate
  if (!is_valid_real(param[1], &strain_rate_)) {
    PRINT_INPUT_ERROR("strain_rate should be a number.\n");
  }
  if (strain_rate_ <= 0) {
    PRINT_INPUT_ERROR("strain_rate should be positive.\n");
  }

  // Parse direction
  parse_direction(param[2]);

  // Parse optional dump_interval (default: 1000)
  dump_interval_ = 1000;
  if (num_param >= 4) {
    int interval;
    if (!is_valid_int(param[3], &interval)) {
      PRINT_INPUT_ERROR("dump_interval should be an integer.\n");
    }
    if (interval > 0) {
      dump_interval_ = interval;
    }
  }

  // Parse optional output_dir (default: ./)
  if (num_param >= 5 && param[4] != nullptr && strlen(param[4]) > 0) {
    strncpy(output_dir_, param[4], sizeof(output_dir_) - 2);
    output_dir_[sizeof(output_dir_) - 2] = '\0';
    // Ensure trailing slash
    size_t len = strlen(output_dir_);
    if (len > 0 && output_dir_[len - 1] != '/') {
      output_dir_[len] = '/';
      output_dir_[len + 1] = '\0';
    }
    // Create directory using system command
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", output_dir_);
    int ret = system(mkdir_cmd);
    (void)ret;  // Suppress unused warning
  } else {
    // Default: ./
    output_dir_[0] = '.';
    output_dir_[1] = '/';
    output_dir_[2] = '\0';
  }

  // Parse optional file_prefix (default: shear)
  if (num_param >= 6 && param[5] != nullptr && strlen(param[5]) > 0) {
    strncpy(file_prefix_, param[5], sizeof(file_prefix_) - 1);
    file_prefix_[sizeof(file_prefix_) - 1] = '\0';
  } else {
    strcpy(file_prefix_, "shear");
  }

  active_ = true;
  current_step_ = 0;

  printf("    Strain rate: %g per step\n", strain_rate_);
  printf("    Direction: %s\n", direction_);
  printf("    Shear component: %c%c\n", 'x' + shear_i_, 'x' + shear_j_);
  printf("    Effective strain per step: %g\n", strain_rate_ * strain_sign_);
  printf("    Dump interval: %d steps\n", dump_interval_);
  printf("    Output directory: %s\n", output_dir_);
  printf("    File prefix: %s\n", file_prefix_);
  printf("---------------------------------------------------------------\n");
}

void Shear::apply(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  double step_strain)
{
  if (!active_) return;

  const int N = position_per_atom.size() / 3;
  
  int idx_ij = shear_i_ * 3 + shear_j_;
  int idx_jj = shear_j_ * 3 + shear_j_;
  
  box.cpu_h[idx_ij] += step_strain * box.cpu_h[idx_jj];
  box.get_inverse();

  double* x = position_per_atom.data();
  double* y = position_per_atom.data() + N;
  double* z = position_per_atom.data() + 2 * N;
  
  gpu_apply_shear_to_atoms<<<(N - 1) / 128 + 1, 128>>>(
    N, shear_i_, shear_j_, step_strain, x, y, z
  );
  GPU_CHECK_KERNEL
}

void Shear::check_and_dump(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type)
{
  if (!active_) return;
  
  current_step_++;
  
  if (current_step_ > 0 && current_step_ % dump_interval_ == 0) {
    dump_structure(current_step_, box, position_per_atom, type);
  }
}

void Shear::dump_structure(
  int step,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type)
{
  if (!active_) return;
  
  const int N = type.size();
  
  std::vector<double> pos_cpu(3 * N);
  std::vector<int> type_cpu(N);
  position_per_atom.copy_to_host(pos_cpu.data());
  type.copy_to_host(type_cpu.data());
  
  // Build filename: output_dir/prefix_step_XXXX.xyz
  char filename[512];
  snprintf(filename, sizeof(filename), "%s%s_step_%04d.xyz", 
           output_dir_, file_prefix_, step);
  
  FILE* fid = fopen(filename, "w");
  if (fid == NULL) {
    printf("Warning: Failed to open %s for writing.\n", filename);
    return;
  }
  
  // Extended XYZ with Lattice (column-major format)
  fprintf(fid, "%d\n", N);
  fprintf(fid, "pbc=\"T T T\" Lattice=\"%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\"\n",
    box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
    box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
    box.cpu_h[2], box.cpu_h[5], box.cpu_h[8]);
  
  for (int n = 0; n < N; ++n) {
    int atom_type = type_cpu[n];
    const char* symbol = (atom_type == 0) ? "Cu" : "Zr";
    
    fprintf(fid, "%s %.10f %.10f %.10f\n",
      symbol,
      pos_cpu[n],
      pos_cpu[n + N],
      pos_cpu[n + 2 * N]);
  }
  
  fclose(fid);
  double strain_pct = step * strain_rate_ * 100;
  printf("    Shear: Saved %s (step %d, strain %.2f%%)\n", filename, step, strain_pct);
}
