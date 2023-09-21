#pragma once
#include <cuda_runtime.h>
#include "math.cuh"
#include "vectorize_scene.cuh"

//Globals
__global__ void GenerateSolverStates(const ATOMIC_SolverState* d_SolverStates, int StatesSize, ATOMIC_SolverState* d_NewSolverStates);
__global__ void MarkInvalidDuplicatesFromGlobal(ATOMIC_SolverState* StatesToMark, int N_StatesToMark, ATOMIC_SolverState* StatesFind, int N_StatesFind);