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
NEB (Nudged Elastic Band) implementation

This implements the Nudged Elastic Band method for finding transition states
and minimum energy paths between two configurations.

The algorithm follows:
1. Create a chain of replicas (images) between initial and final states
2. For each replica (except endpoints), compute:
   - The tangent vector along the path
   - The perpendicular component of the true force
   - The spring force along the tangent
3. Use a minimizer to relax the replicas
4. Optionally use climbing image NEB to find the saddle point

Reference:
[1] Henkelman et al., J. Chem. Phys. 113, 9978 (2000)
[2] Henkelman and Jonsson, J. Chem. Phys. 113, 9980 (2000)
------------------------------------------------------------------------------*/

#include "neb.cuh"
#include "force/force.cuh"
#include "minimize/minimizer_fire.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "model/box.cuh"
#include "model/group.cuh"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// CUDA kernel for copying positions
static __global__ void copy_positions_kernel(
  const int N,
  const double* src,
  double* dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    dst[idx] = src[idx];
  }
}

// CUDA kernel for linear interpolation
static __global__ void interpolate_kernel(
  const int N,
  const double* pos_ini,
  const double* pos_fin,
  double* pos_out,
  const double lambda)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    pos_out[idx] = pos_ini[idx] + lambda * (pos_fin[idx] - pos_ini[idx]);
  }
}

// CUDA kernel for calculating displacement with PBC
static __global__ void calculate_displacement_kernel(
  const int number_of_atoms,
  const double* pos1,
  const double* pos2,
  const double* box,
  double* displacement)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < number_of_atoms * 3) {
    double diff = pos2[idx] - pos1[idx];
    
    // Apply minimum image convention for periodic boundaries
    int dim = idx % 3;
    if (box[dim] > 0) {
      double half_box = box[dim] * 0.5;
      if (diff > half_box) {
        diff -= box[dim];
      } else if (diff < -half_box) {
        diff += box[dim];
      }
    }
    
    displacement[idx] = diff;
  }
}

// CUDA kernel for computing tangent vector (improved tangent method)
static __global__ void compute_tangent_kernel(
  const int number_of_atoms,
  const double* pos_prev,
  const double* pos_curr,
  const double* pos_next,
  const double* box,
  const double V_prev,
  const double V_curr,
  const double V_next,
  double* tangent)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < number_of_atoms * 3) {
    double t1 = 0.0, t2 = 0.0;
    
    // Displacement to previous replica
    double diff1 = pos_curr[idx] - pos_prev[idx];
    int dim = idx % 3;
    if (box[dim] > 0) {
      double half_box = box[dim] * 0.5;
      if (diff1 > half_box) {
        diff1 -= box[dim];
      } else if (diff1 < -half_box) {
        diff1 += box[dim];
      }
    }
    
    // Displacement to next replica
    double diff2 = pos_next[idx] - pos_curr[idx];
    if (box[dim] > 0) {
      double half_box = box[dim] * 0.5;
      if (diff2 > half_box) {
        diff2 -= box[dim];
      } else if (diff2 < -half_box) {
        diff2 += box[dim];
      }
    }
    
    // Improved tangent method (Henkelman et al.)
    if (V_next > V_curr && V_curr > V_prev) {
      // Energy increasing along path
      tangent[idx] = diff2;
    } else if (V_next < V_curr && V_curr < V_prev) {
      // Energy decreasing along path
      tangent[idx] = diff1;
    } else {
      // Energy extremum - use weighted combination
      double V_max = fmax(fabs(V_next - V_curr), fabs(V_prev - V_curr));
      double V_min = fmin(fabs(V_next - V_curr), fabs(V_prev - V_curr));
      
      if (V_next > V_prev) {
        tangent[idx] = diff2 * V_max + diff1 * V_min;
      } else {
        tangent[idx] = diff2 * V_min + diff1 * V_max;
      }
    }
  }
}

// CUDA kernel for normalizing tangent vector
static __global__ void normalize_tangent_kernel(
  const int number_of_atoms,
  double* tangent,
  const double norm)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < number_of_atoms * 3 && norm > 0) {
    tangent[idx] /= norm;
  }
}

// CUDA kernel for computing spring force
static __global__ void compute_spring_force_kernel(
  const int number_of_atoms,
  const double* pos_prev,
  const double* pos_curr,
  const double* pos_next,
  const double* box,
  const double kspring,
  double* spring_force)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < number_of_atoms * 3) {
    double diff_prev = 0.0, diff_next = 0.0;
    int dim = idx % 3;
    
    // Distance to previous replica
    double dp = pos_curr[idx] - pos_prev[idx];
    if (box[dim] > 0) {
      double half_box = box[dim] * 0.5;
      if (dp > half_box) {
        dp -= box[dim];
      } else if (dp < -half_box) {
        dp += box[dim];
      }
    }
    diff_prev = dp;
    
    // Distance to next replica
    double dn = pos_next[idx] - pos_curr[idx];
    if (box[dim] > 0) {
      double half_box = box[dim] * 0.5;
      if (dn > half_box) {
        dn -= box[dim];
      } else if (dn < -half_box) {
        dn += box[dim];
      }
    }
    diff_next = dn;
    
    // Spring force = k * (|R_next - R| - |R - R_prev|) along tangent
    spring_force[idx] = kspring * (diff_next - diff_prev);
  }
}

// CUDA kernel for projecting out parallel component and adding spring force
static __global__ void project_force_kernel(
  const int number_of_atoms,
  const double* tangent,
  const double* spring_force,
  double* force,
  const int is_climbing)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < number_of_atoms * 3) {
    // Compute dot product F . tangent
    double F_dot_t = 0.0;
    for (int i = 0; i < number_of_atoms * 3; ++i) {
      F_dot_t += force[i] * tangent[i];
    }
    
    if (is_climbing) {
      // Climbing image: reverse parallel component
      force[idx] = force[idx] - 2.0 * F_dot_t * tangent[idx];
    } else {
      // Regular NEB: remove parallel component and add spring force
      force[idx] = force[idx] - F_dot_t * tangent[idx] + spring_force[idx];
    }
  }
}

// CUDA kernel for computing dot product (partial)
static __global__ void dot_product_kernel(
  const int N,
  const double* a,
  const double* b,
  double* result)
{
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  double sum = 0.0;
  if (idx < N) {
    sum = a[idx] * b[idx];
  }
  sdata[tid] = sum;
  __syncthreads();
  
  // Reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    result[blockIdx.x] = sdata[0];
  }
}

// Host function to compute dot product on GPU
static double gpu_dot_product(
  const int N,
  const double* d_a,
  const double* d_b,
  double* d_temp)
{
  const int block_size = 256;
  const int num_blocks = (N + block_size - 1) / block_size;
  
  dot_product_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(
    N, d_a, d_b, d_temp);
  GPU_CHECK_KERNEL
  
  // Copy partial results to host and sum
  std::vector<double> h_temp(num_blocks);
  CHECK(gpuMemcpy(h_temp.data(), d_temp, num_blocks * sizeof(double), gpuMemcpyDeviceToHost));
  
  double sum = 0.0;
  for (int i = 0; i < num_blocks; ++i) {
    sum += h_temp[i];
  }
  return sum;
}

// Constructor
NEB::NEB() {}

// Destructor
NEB::~NEB() {}

// Parse NEB command
void NEB::parse_neb(
  const char** param,
  int num_param,
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  printf("\n------------------------------------------------------------\n");
  printf("Started NEB (Nudged Elastic Band) calculation.\n");
  printf("------------------------------------------------------------\n\n");
  
  // Parse parameters
  // neb <nreplica> <etol> <ftol> <n1steps> <n2steps> <nevery> [keywords]
  if (num_param < 7) {
    PRINT_INPUT_ERROR("neb should have at least 6 parameters.\n");
  }
  
  // Number of replicas
  if (!is_valid_int(param[1], &number_of_replicas_)) {
    PRINT_INPUT_ERROR("Number of replicas should be an integer.\n");
  }
  if (number_of_replicas_ < 3) {
    PRINT_INPUT_ERROR("Number of replicas should be at least 3.\n");
  }
  
  // Energy tolerance
  if (!is_valid_real(param[2], &energy_tolerance_)) {
    PRINT_INPUT_ERROR("Energy tolerance should be a number.\n");
  }
  if (energy_tolerance_ < 0) {
    PRINT_INPUT_ERROR("Energy tolerance should be non-negative.\n");
  }
  
  // Force tolerance
  if (!is_valid_real(param[3], &force_tolerance_)) {
    PRINT_INPUT_ERROR("Force tolerance should be a number.\n");
  }
  if (force_tolerance_ < 0) {
    PRINT_INPUT_ERROR("Force tolerance should be non-negative.\n");
  }
  
  // Number of steps for stage 1
  if (!is_valid_int(param[4], &number_of_steps_1_)) {
    PRINT_INPUT_ERROR("Number of steps for stage 1 should be an integer.\n");
  }
  if (number_of_steps_1_ <= 0) {
    PRINT_INPUT_ERROR("Number of steps for stage 1 should be positive.\n");
  }
  
  // Number of steps for stage 2
  if (!is_valid_int(param[5], &number_of_steps_2_)) {
    PRINT_INPUT_ERROR("Number of steps for stage 2 should be an integer.\n");
  }
  if (number_of_steps_2_ < 0) {
    PRINT_INPUT_ERROR("Number of steps for stage 2 should be non-negative.\n");
  }
  
  // Output frequency
  if (!is_valid_int(param[6], &output_frequency_)) {
    PRINT_INPUT_ERROR("Output frequency should be an integer.\n");
  }
  if (output_frequency_ <= 0) {
    PRINT_INPUT_ERROR("Output frequency should be positive.\n");
  }
  
  // Parse optional keywords
  int iarg = 7;
  while (iarg < num_param) {
    if (strcmp(param[iarg], "spring") == 0) {
      if (iarg + 1 >= num_param) {
        PRINT_INPUT_ERROR("spring keyword requires a value.\n");
      }
      if (!is_valid_real(param[iarg + 1], &spring_constant_)) {
        PRINT_INPUT_ERROR("Spring constant should be a number.\n");
      }
      if (spring_constant_ <= 0) {
        PRINT_INPUT_ERROR("Spring constant should be positive.\n");
      }
      iarg += 2;
    } else if (strcmp(param[iarg], "initial") == 0) {
      if (iarg + 1 >= num_param) {
        PRINT_INPUT_ERROR("initial keyword requires a filename.\n");
      }
      initial_file_ = param[iarg + 1];
      iarg += 2;
    } else if (strcmp(param[iarg], "final") == 0) {
      if (iarg + 1 >= num_param) {
        PRINT_INPUT_ERROR("final keyword requires a filename.\n");
      }
      final_file_ = param[iarg + 1];
      iarg += 2;
    } else if (strcmp(param[iarg], "free_end_ini") == 0) {
      free_end_ini_ = 1;
      if (iarg + 1 < num_param && is_valid_real(param[iarg + 1], &kspring_ini_)) {
        iarg += 2;
      } else {
        kspring_ini_ = spring_constant_;
        iarg += 1;
      }
    } else if (strcmp(param[iarg], "free_end_final") == 0) {
      free_end_final_ = 1;
      if (iarg + 1 < num_param && is_valid_real(param[iarg + 1], &kspring_final_)) {
        iarg += 2;
      } else {
        kspring_final_ = spring_constant_;
        iarg += 1;
      }
    } else if (strcmp(param[iarg], "mode") == 0) {
      if (iarg + 1 >= num_param) {
        PRINT_INPUT_ERROR("mode keyword requires a value.\n");
      }
      if (strcmp(param[iarg + 1], "neighbor") == 0) {
        neb_mode_ = 0;
      } else if (strcmp(param[iarg + 1], "ideal") == 0) {
        neb_mode_ = 1;
      } else if (strcmp(param[iarg + 1], "equal") == 0) {
        neb_mode_ = 2;
      } else {
        PRINT_INPUT_ERROR("Invalid NEB mode. Use neighbor, ideal, or equal.\n");
      }
      iarg += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown NEB keyword.\n");
    }
  }
  
  // Print parameters
  printf("NEB Parameters:\n");
  printf("  Number of replicas: %d\n", number_of_replicas_);
  printf("  Energy tolerance: %g eV\n", energy_tolerance_);
  printf("  Force tolerance: %g eV/Angstrom\n", force_tolerance_);
  printf("  Stage 1 steps (regular NEB): %d\n", number_of_steps_1_);
  printf("  Stage 2 steps (climbing NEB): %d\n", number_of_steps_2_);
  printf("  Output frequency: %d\n", output_frequency_);
  printf("  Spring constant: %g eV/Angstrom^2\n", spring_constant_);
  if (free_end_ini_) {
    printf("  Free end initial: yes (kspring = %g)\n", kspring_ini_);
  }
  if (free_end_final_) {
    printf("  Free end final: yes (kspring = %g)\n", kspring_final_);
  }
  printf("\n");
  
  const int number_of_atoms = type.size();
  
  // Allocate memory
  allocate_memory(number_of_atoms);
  
  // Read initial and final configurations
  read_configurations(box, position_per_atom, type, group);
  
  // Initialize replica positions by linear interpolation
  interpolate_replicas(number_of_atoms);
  
  // Stage 1: Regular NEB
  printf("\n------------------------------------------------------------\n");
  printf("Stage 1: Regular NEB minimization\n");
  printf("------------------------------------------------------------\n");
  
  for (int step = 0; step < number_of_steps_1_; ++step) {
    // Process each replica
    double max_force = 0.0;
    
    for (int replica = 0; replica < number_of_replicas_; ++replica) {
      // Skip endpoints unless free ends are enabled
      if (replica == 0 && !free_end_ini_) continue;
      if (replica == number_of_replicas_ - 1 && !free_end_final_) continue;
      
      // Compute NEB forces for this replica
      compute_neb_forces(
        force, box, type, group, 
        potential_per_atom, virial_per_atom, 
        replica);
      
      // Compute force norm
      double f_norm = compute_force_norm(replica, number_of_atoms);
      if (f_norm > max_force) {
        max_force = f_norm;
      }
    }
    
    // Output progress
    if (step % output_frequency_ == 0) {
      output_progress(step, 1, number_of_atoms);
    }
    
    // Check convergence
    if (max_force < force_tolerance_) {
      printf("\nStage 1 converged at step %d (max force = %g)\n", step, max_force);
      break;
    }
  }
  
  // Find the climbing replica (highest energy)
  find_climbing_replica();
  
  // Stage 2: Climbing NEB (if requested)
  if (number_of_steps_2_ > 0 && climber_ >= 0) {
    printf("\n------------------------------------------------------------\n");
    printf("Stage 2: Climbing NEB minimization\n");
    printf("  Climbing replica: %d\n", climber_ + 1);
    printf("------------------------------------------------------------\n");
    
    for (int step = 0; step < number_of_steps_2_; ++step) {
      double max_force = 0.0;
      
      for (int replica = 0; replica < number_of_replicas_; ++replica) {
        // Skip endpoints
        if (replica == 0 || replica == number_of_replicas_ - 1) continue;
        
        // Compute NEB forces (with climbing for the highest energy replica)
        compute_neb_forces(
          force, box, type, group,
          potential_per_atom, virial_per_atom,
          replica);
        
        double f_norm = compute_force_norm(replica, number_of_atoms);
        if (f_norm > max_force) {
          max_force = f_norm;
        }
      }
      
      if (step % output_frequency_ == 0) {
        output_progress(step, 2, number_of_atoms);
      }
      
      if (max_force < force_tolerance_) {
        printf("\nStage 2 converged at step %d (max force = %g)\n", step, max_force);
        break;
      }
    }
  }
  
  // Save final path
  save_final_path("neb_final_path.xyz", number_of_atoms);
  
  printf("\n------------------------------------------------------------\n");
  printf("NEB calculation completed.\n");
  printf("------------------------------------------------------------\n\n");
}

// Allocate memory
void NEB::allocate_memory(int number_of_atoms)
{
  const int num_elements = number_of_atoms * 3;
  
  // Host arrays
  h_position_initial_.resize(num_elements);
  h_position_final_.resize(num_elements);
  h_positions_all_.resize(number_of_replicas_ * num_elements);
  h_forces_all_.resize(number_of_replicas_ * num_elements);
  h_energies_all_.resize(number_of_replicas_);
  h_tangents_.resize(number_of_replicas_ * num_elements);
  h_spring_forces_.resize(number_of_replicas_ * num_elements);
  
  // Device arrays
  d_position_initial_.resize(num_elements);
  d_position_final_.resize(num_elements);
  d_positions_replica_.resize(num_elements);
  d_forces_replica_.resize(num_elements);
  d_tangent_.resize(num_elements);
  d_spring_force_.resize(num_elements);
}

// Read initial and final configurations
void NEB::read_configurations(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group)
{
  // For now, use current configuration as initial
  // and read final from file if provided
  
  const int number_of_atoms = type.size();
  const int num_elements = number_of_atoms * 3;
  
  // Copy current positions as initial configuration
  CHECK(gpuMemcpy(h_position_initial_.data(), 
                  position_per_atom.data(), 
                  num_elements * sizeof(double), 
                  gpuMemcpyDeviceToHost));
  
  // Read final configuration from file
  if (final_file_.empty()) {
    PRINT_INPUT_ERROR("NEB requires a final configuration file. Use 'final' keyword.\n");
  }
  
  printf("Reading final configuration from %s\n", final_file_.c_str());
  
  std::ifstream file(final_file_);
  if (!file.is_open()) {
    PRINT_INPUT_ERROR("Cannot open final configuration file.\n");
  }
  
  std::string line;
  int num_atoms;
  
  // Read number of atoms
  std::getline(file, line);
  std::istringstream iss(line);
  iss >> num_atoms;
  
  if (num_atoms != number_of_atoms) {
    PRINT_INPUT_ERROR("Number of atoms in final configuration does not match.\n");
  }
  
  // Skip comment line
  std::getline(file, line);
  
  // Read atom positions
  for (int i = 0; i < num_atoms; ++i) {
    std::getline(file, line);
    std::istringstream atom_iss(line);
    std::string element;
    double x, y, z;
    atom_iss >> element >> x >> y >> z;
    
    h_position_final_[i * 3] = x;
    h_position_final_[i * 3 + 1] = y;
    h_position_final_[i * 3 + 2] = z;
  }
  
  file.close();
  
  // Copy to device
  CHECK(gpuMemcpy(d_position_initial_.data(),
                  h_position_initial_.data(),
                  num_elements * sizeof(double),
                  gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_position_final_.data(),
                  h_position_final_.data(),
                  num_elements * sizeof(double),
                  gpuMemcpyHostToDevice));
}

// Linear interpolation to initialize replicas
void NEB::interpolate_replicas(int number_of_atoms)
{
  const int num_elements = number_of_atoms * 3;
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;
  
  // First replica = initial configuration
  for (int i = 0; i < num_elements; ++i) {
    h_positions_all_[i] = h_position_initial_[i];
  }
  
  // Last replica = final configuration
  for (int i = 0; i < num_elements; ++i) {
    h_positions_all_[(number_of_replicas_ - 1) * num_elements + i] = h_position_final_[i];
  }
  
  // Intermediate replicas by linear interpolation
  for (int replica = 1; replica < number_of_replicas_ - 1; ++replica) {
    double lambda = static_cast<double>(replica) / (number_of_replicas_ - 1);
    
    double* pos_out = &h_positions_all_[replica * num_elements];
    for (int i = 0; i < num_elements; ++i) {
      pos_out[i] = h_position_initial_[i] + lambda * (h_position_final_[i] - h_position_initial_[i]);
    }
  }
  
  printf("Initialized %d replicas by linear interpolation.\n", number_of_replicas_);
}

// Compute NEB forces for a replica
void NEB::compute_neb_forces(
  Force& force,
  Box& box,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& virial_per_atom,
  int replica_index)
{
  const int number_of_atoms = type.size();
  const int num_elements = number_of_atoms * 3;
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;
  
  // Copy replica positions to device
  CHECK(gpuMemcpy(d_positions_replica_.data(),
                  &h_positions_all_[replica_index * num_elements],
                  num_elements * sizeof(double),
                  gpuMemcpyHostToDevice));
  
  // Compute true forces using the potential
  force.compute(
    box,
    d_positions_replica_,
    type,
    group,
    potential_per_atom,
    d_forces_replica_,
    virial_per_atom);
  
  // Store energy
  double total_energy = 0.0;
  std::vector<double> h_potential(number_of_atoms);
  CHECK(gpuMemcpy(h_potential.data(),
                  potential_per_atom.data(),
                  number_of_atoms * sizeof(double),
                  gpuMemcpyDeviceToHost));
  for (int i = 0; i < number_of_atoms; ++i) {
    total_energy += h_potential[i];
  }
  h_energies_all_[replica_index] = total_energy;
  
  // Store forces
  CHECK(gpuMemcpy(&h_forces_all_[replica_index * num_elements],
                  d_forces_replica_.data(),
                  num_elements * sizeof(double),
                  gpuMemcpyDeviceToHost));
  
  // For endpoints without free ends, no modification needed
  if (replica_index == 0 && !free_end_ini_) return;
  if (replica_index == number_of_replicas_ - 1 && !free_end_final_) return;
  
  // Calculate tangent vector
  calculate_tangent(replica_index, number_of_atoms);
  
  // Calculate spring force
  apply_spring_force(replica_index, number_of_atoms);
  
  // Project out parallel component and add spring force
  // For climbing replica, invert the parallel component
  int is_climbing = (replica_index == climber_) ? 1 : 0;
  
  // Copy tangent and spring force to device
  CHECK(gpuMemcpy(d_tangent_.data(),
                  &h_tangents_[replica_index * num_elements],
                  num_elements * sizeof(double),
                  gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_spring_force_.data(),
                  &h_spring_forces_[replica_index * num_elements],
                  num_elements * sizeof(double),
                  gpuMemcpyHostToDevice));
  
  // Apply NEB force projection
  project_force_kernel<<<num_blocks, block_size>>>(
    number_of_atoms,
    d_tangent_.data(),
    d_spring_force_.data(),
    d_forces_replica_.data(),
    is_climbing);
  GPU_CHECK_KERNEL
  
  // Copy modified forces back
  CHECK(gpuMemcpy(&h_forces_all_[replica_index * num_elements],
                  d_forces_replica_.data(),
                  num_elements * sizeof(double),
                  gpuMemcpyDeviceToHost));
}

// Calculate tangent vector using improved tangent method
void NEB::calculate_tangent(int replica_index, int number_of_atoms)
{
  const int num_elements = number_of_atoms * 3;
  
  // For endpoints, use simple tangent
  if (replica_index == 0) {
    // First replica: tangent points to next replica
    for (int i = 0; i < num_elements; ++i) {
      h_tangents_[replica_index * num_elements + i] = 
        h_positions_all_[(replica_index + 1) * num_elements + i] - 
        h_positions_all_[replica_index * num_elements + i];
    }
    return;
  }
  
  if (replica_index == number_of_replicas_ - 1) {
    // Last replica: tangent points from previous replica
    for (int i = 0; i < num_elements; ++i) {
      h_tangents_[replica_index * num_elements + i] = 
        h_positions_all_[replica_index * num_elements + i] - 
        h_positions_all_[(replica_index - 1) * num_elements + i];
    }
    return;
  }
  
  // Interior replicas: use improved tangent method
  double V_prev = h_energies_all_[replica_index - 1];
  double V_curr = h_energies_all_[replica_index];
  double V_next = h_energies_all_[replica_index + 1];
  
  double* tangent = &h_tangents_[replica_index * num_elements];
  
  if (V_next > V_curr && V_curr > V_prev) {
    // Energy increasing: use forward tangent
    for (int i = 0; i < num_elements; ++i) {
      tangent[i] = h_positions_all_[(replica_index + 1) * num_elements + i] - 
                   h_positions_all_[replica_index * num_elements + i];
    }
  } else if (V_next < V_curr && V_curr < V_prev) {
    // Energy decreasing: use backward tangent
    for (int i = 0; i < num_elements; ++i) {
      tangent[i] = h_positions_all_[replica_index * num_elements + i] - 
                   h_positions_all_[(replica_index - 1) * num_elements + i];
    }
  } else {
    // Energy extremum: use weighted combination
    double V_max = std::max(std::abs(V_next - V_curr), std::abs(V_prev - V_curr));
    double V_min = std::min(std::abs(V_next - V_curr), std::abs(V_prev - V_curr));
    
    double* t1 = &h_positions_all_[replica_index * num_elements];
    double* t2_prev = &h_positions_all_[(replica_index - 1) * num_elements];
    double* t2_next = &h_positions_all_[(replica_index + 1) * num_elements];
    
    if (V_next > V_prev) {
      for (int i = 0; i < num_elements; ++i) {
        double tau1 = t2_next[i] - t1[i];
        double tau2 = t1[i] - t2_prev[i];
        tangent[i] = tau1 * V_max + tau2 * V_min;
      }
    } else {
      for (int i = 0; i < num_elements; ++i) {
        double tau1 = t2_next[i] - t1[i];
        double tau2 = t1[i] - t2_prev[i];
        tangent[i] = tau1 * V_min + tau2 * V_max;
      }
    }
  }
  
  // Normalize tangent
  double norm = 0.0;
  for (int i = 0; i < num_elements; ++i) {
    norm += tangent[i] * tangent[i];
  }
  norm = std::sqrt(norm);
  
  if (norm > 0) {
    for (int i = 0; i < num_elements; ++i) {
      tangent[i] /= norm;
    }
  }
}

// Apply spring force
void NEB::apply_spring_force(int replica_index, int number_of_atoms)
{
  const int num_elements = number_of_atoms * 3;
  double* spring_force = &h_spring_forces_[replica_index * num_elements];
  
  // Calculate distances to neighboring replicas
  double dist_prev = 0.0, dist_next = 0.0;
  
  for (int i = 0; i < num_elements; ++i) {
    double diff_prev = h_positions_all_[replica_index * num_elements + i] - 
                       h_positions_all_[(replica_index - 1) * num_elements + i];
    double diff_next = h_positions_all_[(replica_index + 1) * num_elements + i] - 
                       h_positions_all_[replica_index * num_elements + i];
    
    dist_prev += diff_prev * diff_prev;
    dist_next += diff_next * diff_next;
  }
  
  dist_prev = std::sqrt(dist_prev);
  dist_next = std::sqrt(dist_next);
  
  // Spring force along tangent
  double spring_magnitude = spring_constant_ * (dist_next - dist_prev);
  
  double* tangent = &h_tangents_[replica_index * num_elements];
  for (int i = 0; i < num_elements; ++i) {
    spring_force[i] = spring_magnitude * tangent[i];
  }
}

// Compute force norm for convergence check
double NEB::compute_force_norm(int replica_index, int number_of_atoms)
{
  const int num_elements = number_of_atoms * 3;
  double* forces = &h_forces_all_[replica_index * num_elements];
  
  double norm = 0.0;
  for (int i = 0; i < num_elements; ++i) {
    norm += forces[i] * forces[i];
  }
  
  return std::sqrt(norm / number_of_atoms);
}

// Find the climbing replica (highest energy)
void NEB::find_climbing_replica()
{
  double max_energy = h_energies_all_[0];
  climber_ = 0;
  
  for (int i = 1; i < number_of_replicas_; ++i) {
    if (h_energies_all_[i] > max_energy) {
      max_energy = h_energies_all_[i];
      climber_ = i;
    }
  }
  
  printf("\nHighest energy replica: %d (E = %g eV)\n", climber_ + 1, max_energy);
}

// Output progress
void NEB::output_progress(int step, int stage, int number_of_atoms)
{
  printf("Stage %d, Step %d:\n", stage, step);
  printf("  Replica  Energy (eV)    Max Force (eV/A)\n");
  
  for (int i = 0; i < number_of_replicas_; ++i) {
    double f_norm = compute_force_norm(i, number_of_atoms);
    printf("  %3d      %12.6f    %12.6f\n", i + 1, h_energies_all_[i], f_norm);
  }
  printf("\n");
}

// Save final path to file
void NEB::save_final_path(const char* filename, int number_of_atoms)
{
  std::ofstream file(filename);
  if (!file.is_open()) {
    printf("Warning: Could not open %s for writing.\n", filename);
    return;
  }
  
  for (int replica = 0; replica < number_of_replicas_; ++replica) {
    file << number_of_atoms << "\n";
    file << "Replica " << replica + 1 << " Energy = " << h_energies_all_[replica] << " eV\n";
    
    for (int i = 0; i < number_of_atoms; ++i) {
      file << "X "  // Placeholder element
           << h_positions_all_[replica * number_of_atoms * 3 + i * 3] << " "
           << h_positions_all_[replica * number_of_atoms * 3 + i * 3 + 1] << " "
           << h_positions_all_[replica * number_of_atoms * 3 + i * 3 + 2] << "\n";
    }
  }
  
  file.close();
  printf("\nFinal NEB path saved to %s\n", filename);
}

// Minimize a single replica using FIRE
void NEB::minimize_replica(
  Force& force,
  Box& box,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& virial_per_atom,
  int replica_index,
  int max_steps)
{
  // This is a placeholder for more sophisticated minimization
  // For now, we use simple steepest descent in the main loop
}
