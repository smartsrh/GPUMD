/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.

    Dump per-atom stress to a file.
*/

#pragma once
#include "measure.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Dump_Stress : public Property {
public:
  Dump_Stress(const char** param, int num_param);
  ~Dump_Stress() {};

  void parse(const char** param, int num_param);

  void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  void process(
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
    Force& force);

  void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

private:
  int dump_interval_;
  char output_dir_[256];
};
