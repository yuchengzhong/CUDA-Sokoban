#pragma once
#include <cuda_runtime.h>
#include "math.cuh"
#include "vectorize_scene.cuh"
#include "timer.h"

struct GPU_Solver
{
public:
    static vector<ATOMIC_Steps> GPU_Solver::Solve(const ATOMIC_Scene& InitialScene, const STATIC_SceneBlock& SceneBlock, bool ShortestOnly, bool Debug = false);
};