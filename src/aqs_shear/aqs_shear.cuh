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
------------------------------------------------------------------------------*/

#pragma once
#include "force/force.cuh"
#include "minimize/minimizer_sd.cuh"
#include "minimize/minimizer_fire.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class AQSShear {
public:
  enum class MinimizerType { SD, FIRE };
  enum class ShearMode { PURE, SIMPLE };

  struct Parameters {
    std::string direction;
    double strain_per_step;
    double total_strain;
    int total_steps;
    MinimizerType minimizer;
    ShearMode shear_mode;
    double force_tolerance;
    int max_minimize_steps;
    int dump_frequency;
    std::string output_prefix;
    std::string output_dir;      // Output directory for all files
    std::string stress_file;
    std::string atom_stress_file;
    bool initial_relax;
    bool dump_atom_stress;
    std::vector<std::string> element_names;  // Element symbols for XYZ output
  };

  AQSShear();
  
  void parse(const char** param, int num_param);
  
  void compute(
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom
  );

private:
  Parameters params_;
  
  void parse_direction(const char* direction_str);
  void parse_minimizer(const char* minimizer_str);
  void parse_shear_mode(const char* mode_str);
  void parse_optional_params(const char** param, int num_param, int start_idx);
  
  // Fixed: Apply shear with consistent box and atom transformation
  void apply_shear(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    double strain_increment
  );
  
  void minimize_energy(
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom
  );
  
  void compute_stress_tensor(
    GPU_Vector<double>& virial_per_atom,
    Box& box,
    double stress[6]
  );
  
  void dump_structure(
    const std::string& filename,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<std::string>& element_names
  );
  
  void dump_atom_stress(
    const std::string& filename,
    int step,
    double strain,
    GPU_Vector<double>& virial_per_atom,
    Box& box,
    GPU_Vector<int>& type
  );
  
  void dump_stress_header();
  void dump_stress_data(int step, double strain, double stress[6]);
  
  int get_deform_components(int& i, int& j);
  double get_strain_sign();
};
