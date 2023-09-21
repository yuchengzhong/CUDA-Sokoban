#include "solver_gpu_base.cuh"
#include <thrust/copy.h>
#include <thrust/reduce.h>
//Scan
template <typename Predicate>
thrust::device_vector<ATOMIC_SolverState> Scan(const thrust::device_vector<ATOMIC_SolverState>& NewSolverStates, Predicate Pred)
{
    thrust::device_vector<ATOMIC_SolverState> FilteredSolverStates(NewSolverStates.size());
    auto End = thrust::copy_if
    (
        NewSolverStates.begin(),
        NewSolverStates.end(),
        FilteredSolverStates.begin(),
        Pred
    );
    FilteredSolverStates.resize(thrust::distance(FilteredSolverStates.begin(), End));
    return FilteredSolverStates;
}
template thrust::device_vector<ATOMIC_SolverState> Scan<IsSolverStateValid>(const thrust::device_vector<ATOMIC_SolverState>&, IsSolverStateValid); // let compiler know
template thrust::device_vector<ATOMIC_SolverState> Scan<IsSolverStateWin>(const thrust::device_vector<ATOMIC_SolverState>&, IsSolverStateWin);

thrust::device_vector<ATOMIC_Steps> ExtractSteps(const thrust::device_vector<ATOMIC_SolverState>& SolverStates)
{
    thrust::device_vector<ATOMIC_Steps> ExtractedSteps(SolverStates.size());
    thrust::transform
    (
        SolverStates.begin(),
        SolverStates.end(),
        ExtractedSteps.begin(),
        GetSteps()
    );
    return ExtractedSteps;
}