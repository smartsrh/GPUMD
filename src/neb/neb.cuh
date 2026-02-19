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

#pragma once

#include "utilities/gpu_vector.cuh"
#include <vector>
#include <string>

class Force;
class Box;
class Group;

// NEB (Nudged Elastic Band) method for finding transition states
// This implementation follows the LAMMPS NEB algorithm
class NEB
{
public:
  NEB();
  ~NEB();

  void parse_neb(
    const char** param,
    int num_param,
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

private:
  // NEB parameters
  int number_of_replicas_ = 3;      // Number of replicas (images)
  int number_of_steps_1_ = 1000;    // Number of steps for first stage (regular NEB)
  int number_of_steps_2_ = 1000;    // Number of steps for second stage (climbing NEB)
  int output_frequency_ = 10;       // Output frequency
  double energy_tolerance_ = 1.0e-6; // Energy tolerance for convergence
  double force_tolerance_ = 1.0e-4;  // Force tolerance for convergence (eV/Angstrom)
  double spring_constant_ = 1.0;     // Spring constant between replicas (eV/Angstrom^2)
  
  // NEB modes
  int neb_mode_ = 0;   // 0 = NEIGHBOR (default), 1 = IDEAL, 2 = EQUAL
  int free_end_ini_ = 0;   // Free end for initial replica
  int free_end_final_ = 0; // Free end for final replica
  double kspring_ini_ = 0.0;   // Spring constant for free end initial
  double kspring_final_ = 0.0; // Spring constant for free end final
  
  // Climbing image
  int climber_ = -1;   // Index of climbing replica (-1 = no climbing)
  
  // File names for initial and final configurations
  std::string initial_file_ = "";
  std::string final_file_ = "";
  
  // Internal arrays for NEB calculation (host)
  std::vector<double> h_position_initial_;  // Initial configuration
  std::vector<double> h_position_final_;    // Final configuration
  std::vector<double> h_positions_all_;     // All replica positions [replica][atom*3]
  std::vector<double> h_forces_all_;        // All replica forces [replica][atom*3]
  std::vector<double> h_energies_all_;      // All replica energies [replica]
  std::vector<double> h_tangents_;          // Tangents for each replica
  std::vector<double> h_spring_forces_;     // Spring forces
  
  // Device arrays
  GPU_Vector<double> d_position_initial_;
  GPU_Vector<double> d_position_final_;
  GPU_Vector<double> d_positions_replica_;  // Positions for current replica
  GPU_Vector<double> d_forces_replica_;     // Forces for current replica
  GPU_Vector<double> d_tangent_;            // Tangent vector
  GPU_Vector<double> d_spring_force_;       // Spring force
  
  // Private methods
  void allocate_memory(int number_of_atoms);
  void read_configurations(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group);
  void initialize_replicas(
    Box& box,
    GPU_Vector<double>& position_per_atom);
  void compute_neb_forces(
    Force& force,
    Box& box,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& virial_per_atom,
    int replica_index);
  void interpolate_replicas(int number_of_atoms);
  double calculate_path_length(int replica_index, int number_of_atoms);
  void calculate_tangent(int replica_index, int number_of_atoms);
  void apply_spring_force(int replica_index, int number_of_atoms);
  void apply_climbing_force(int replica_index, int number_of_atoms);
  void update_replica_positions(int replica_index, int number_of_atoms, double step_size);
  double compute_force_norm(int replica_index, int number_of_atoms);
  void find_climbing_replica();
  void output_progress(int step, int stage, int number_of_atoms);
  void save_final_path(const char* filename, int number_of_atoms);
  
  // Minimization for NEB
  void minimize_replica(
    Force& force,
    Box& box,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& virial_per_atom,
    int replica_index,
    int max_steps);
};
