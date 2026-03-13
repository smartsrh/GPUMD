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
Shear deformation with temperature control
Similar to 'deform' but for shear instead of uniaxial
Requires NPT or NVT ensemble (like deform)
------------------------------------------------------------------------------*/

#pragma once
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"

class Shear {
public:
  Shear();
  
  void parse(const char** param, int num_param);
  
  // Apply shear deformation for one step (called during MD run)
  void apply(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    double step_strain
  );
  
  // Getters for integrate to check if shear is active
  bool is_active() const { return active_; }
  double get_strain_rate() const { return strain_rate_; }
  int get_shear_i() const { return shear_i_; }
  int get_shear_j() const { return shear_j_; }
  double get_strain_sign() const { return strain_sign_; }
  int get_dump_interval() const { return dump_interval_; }
  const char* get_output_dir() const { return output_dir_; }
  const char* get_file_prefix() const { return file_prefix_; }
  
  // Dump structure to XYZ file
  void dump_structure(
    int step,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type);

  // Check and dump if needed
  void check_and_dump(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type);

private:
  bool active_;
  double strain_rate_;      // Strain per step (dimensionless)
  char direction_[8];       // "xy", "xz", "yz", etc.
  int shear_i_;            // First index for shear (0=x, 1=y, 2=z)
  int shear_j_;            // Second index for shear
  double strain_sign_;     // +1 or -1 for direction
  
  // Output control
  int dump_interval_;       // Output XYZ every N steps (default: 1000)
  char output_dir_[256];    // Output directory (default: ./)
  char file_prefix_[64];    // File prefix for outputs (default: shear)
  int current_step_;        // Current MD step counter
  
  void parse_direction(const char* direction_str);
};
