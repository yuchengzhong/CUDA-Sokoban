#pragma once
#include <cuda_runtime.h>
#include "math.cuh"
#include "vectorize_scene.cuh"
// Thrust
#include <thrust/device_vector.h>
// closure for thrust
struct IsSolverStateValid
{
    __device__ bool operator()(ATOMIC_SolverState State) const
    {
        return State.ValidState;
    }
};
struct IsSolverStateWin
{
    __device__ bool operator()(ATOMIC_SolverState State) const
    {
        return State.WinState;
    }
};
struct GetSteps
{
    __device__ ATOMIC_Steps operator()(ATOMIC_SolverState State) const
    {
        return State.StepState;
    }
};
thrust::device_vector<ATOMIC_Steps> ExtractSteps(const thrust::device_vector<ATOMIC_SolverState>& SolverStates);

template <typename Predicate>
thrust::device_vector<ATOMIC_SolverState> Scan(const thrust::device_vector<ATOMIC_SolverState>& NewSolverStates, Predicate Pred);