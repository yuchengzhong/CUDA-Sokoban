#pragma once
#include <cuda_runtime.h>
#include "math.cuh"
#include "vectorize_scene.cuh"
// Thrust
#include <thrust/device_vector.h>
// closure for thrust
struct IsSolverStateValid
{
    __device__ __forceinline__ bool operator()(ATOMIC_SolverState State) const
    {
        return State.ValidState;
    }
};
struct IsSolverStateWin
{
    __device__ __forceinline__ bool operator()(ATOMIC_SolverState State) const
    {
        return State.WinState;
    }
};
struct GetSteps
{
    __device__ __forceinline__ ATOMIC_Steps operator()(ATOMIC_SolverState State) const
    {
        return State.StepState;
    }
};
thrust::device_vector<ATOMIC_Steps> ExtractSteps(const thrust::device_vector<ATOMIC_SolverState>& SolverStates, const size_t N);

template <typename Predicate>
size_t Scan(thrust::device_vector<ATOMIC_SolverState>& NewSolverStates, const size_t N, Predicate Pred);