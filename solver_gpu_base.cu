#include "solver_gpu_base.cuh"
#include <thrust/copy.h>
#include <thrust/reduce.h>
//Scan
template <typename Predicate>
size_t Scan(thrust::device_vector<ATOMIC_SolverState>& NewSolverStates, const size_t N, Predicate Pred)
{
    auto End = thrust::copy_if
    (
        NewSolverStates.begin(),
        NewSolverStates.begin() + N,
        NewSolverStates.begin(),
        Pred
    );
    size_t NewSize = End - NewSolverStates.begin();
    return NewSize;
}
template size_t Scan<IsSolverStateValid>(thrust::device_vector<ATOMIC_SolverState>&, const size_t, IsSolverStateValid); // let compiler know
template size_t Scan<IsSolverStateWin>(thrust::device_vector<ATOMIC_SolverState>&, const size_t, IsSolverStateWin);

thrust::device_vector<ATOMIC_Steps> ExtractSteps(const thrust::device_vector<ATOMIC_SolverState>& SolverStates, const size_t N)
{
    thrust::device_vector<ATOMIC_Steps> ExtractedSteps(N);
    thrust::transform
    (
        SolverStates.begin(),
        SolverStates.begin() + N,
        ExtractedSteps.begin(),
        GetSteps()
    );
    return ExtractedSteps;
}