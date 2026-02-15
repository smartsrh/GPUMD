/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.

    Dump per-atom stress to a file.
    Usage: dump_stress <interval> [output_dir]
    
    Output format (per step):
    step atom_id s_xx s_yy s_zz s_xy s_xz s_yz
*/

#include "dump_stress.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

Dump_Stress::Dump_Stress(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "dump_stress";
}

void Dump_Stress::parse(const char** param, int num_param)
{
  if (num_param < 2 || num_param > 3) {
    PRINT_INPUT_ERROR("dump_stress requires 1-2 parameters: interval [output_dir]");
  }

  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }

  // Parse optional output directory
  if (num_param >= 3) {
    strncpy(output_dir_, param[2], sizeof(output_dir_) - 1);
    output_dir_[sizeof(output_dir_) - 1] = '\0';
    size_t len = strlen(output_dir_);
    if (len > 0 && output_dir_[len - 1] != '/') {
      output_dir_[len] = '/';
      output_dir_[len + 1] = '\0';
    }
  } else {
    strcpy(output_dir_, "./");
  }

  printf("Dump per-atom stress every %d steps to %s\n", dump_interval_, output_dir_);
}

void Dump_Stress::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  // Create output directory
  char mkdir_cmd[512];
  snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", output_dir_);
  int ret = system(mkdir_cmd);
  (void)ret;
}

void Dump_Stress::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature_target,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& gpu_thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int N = atom.number_of_atoms;

  // Copy virial from GPU to CPU (6 components: xx, yy, zz, yz, xz, xy)
  std::vector<double> virial_cpu(6 * N);
  atom.virial_per_atom.copy_to_host(virial_cpu.data(), 6 * N);

  // Copy type from GPU to CPU
  std::vector<int> type_cpu(N);
  atom.type.copy_to_host(type_cpu.data());

  // Build filename
  char filename[512];
  snprintf(filename, sizeof(filename), "%sstress_step_%04d.dat", output_dir_, step + 1);

  FILE* fid = fopen(filename, "w");
  if (fid == NULL) {
    printf("Warning: Failed to open %s for writing.\n", filename);
    return;
  }

  // Header
  fprintf(fid, "# step atom_id type s_xx s_yy s_zz s_yz s_xz s_xy (GPa)\n");

  // Write per-atom stress (virial is already in GPa in GPUMD)
  // Note: virial = stress * volume, but GPUMD outputs virial directly
  for (int n = 0; n < N; ++n) {
    fprintf(fid, "%d %d %d %.6f %.6f %.6f %.6f %.6f %.6f\n",
      step + 1, n, type_cpu[n],
      virial_cpu[n],           // s_xx
      virial_cpu[n + N],       // s_yy
      virial_cpu[n + 2 * N],   // s_zz
      virial_cpu[n + 3 * N],   // s_yz
      virial_cpu[n + 4 * N],   // s_xz
      virial_cpu[n + 5 * N]    // s_xy
    );
  }

  fclose(fid);
  printf("    Dumped stress to %s\n", filename);
}

void Dump_Stress::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  // Nothing to clean up
}
