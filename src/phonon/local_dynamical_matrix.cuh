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
/* Atom order: i = 0..N-1 matches model.xyz row order; output order idem (util.py 1-to-1). */
#include "utilities/gpu_vector.cuh"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

class Box;
class Group;
class Force;

enum LocalDynamicalMatrixStyle { LOCAL_REGULAR, LOCAL_ESKM };

class LocalDynamicalMatrix
{
public:
  double displacement = 0.00001;
  LocalDynamicalMatrixStyle style = LOCAL_REGULAR;

  void compute(
    Force& force,
    Box& box,
    std::vector<double>& cpu_position_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& mass);

  void parse(const char**, size_t);

protected:
  int number_of_atoms;
  double conversion = 1.0; // unit conversion factor for ESKM mode

  void calculate_matrix(
    Force& force,
    Box& box,
    std::vector<double>& cpu_position_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& mass);

  void displace_atom(
    size_t atom_idx,
    int direction,
    double magnitude,
    GPU_Vector<double>& position_per_atom);

  void write_matrix_row(
    FILE* fp,
    const std::vector<double>& dynmat_row);

  void convert_units_eskm();
};

