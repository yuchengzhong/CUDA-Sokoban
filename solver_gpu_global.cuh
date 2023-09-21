#pragma once
#include <cuda_runtime.h>
#include "math.cuh"
#include "vectorize_scene.cuh"

//Globals
__global__ void GenerateSolverStates(const ATOMIC_SolverState* SolverStates, const STATIC_SceneBlock* SceneBlock, const uint32_t N_SolverStates, ATOMIC_SolverState* d_NewSolverStates);
__global__ void MarkInvalidDuplicatesFromGlobal(ATOMIC_SolverState* StatesToMark, const uint32_t N_StatesToMark, const ATOMIC_SolverState* StatesFind, const uint32_t N_StatesFind);