#pragma once
#include <cuda_runtime.h>
#include "math.cuh"
#include "vectorize_scene.cuh"

struct GPU_Solver
{
public:
    static vector<ATOMIC_Steps> GPU_Solver::Solve(const ATOMIC_Scene& InitialScene, bool ShortestOnly, bool Debug = false);
};