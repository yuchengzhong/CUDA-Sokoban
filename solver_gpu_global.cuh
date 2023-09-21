#pragma once
#include <cuda_runtime.h>
#include "math.cuh"
#include "vectorize_scene.cuh"

//Globals
__global__ void GenerateSolverStates(const ATOMIC_SolverState* SolverStates, const STATIC_SceneBlock* SceneBlock, const int N_SolverStates, ATOMIC_SolverState* d_NewSolverStates);
__global__ void MarkInvalidDuplicatesFromGlobal(ATOMIC_SolverState* StatesToMark, const int N_StatesToMark, const ATOMIC_SolverState* StatesFind, const int N_StatesFind);