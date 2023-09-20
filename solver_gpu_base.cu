#include "solver_gpu_base.cuh"
#include <thrust/copy.h>
#include <thrust/reduce.h>

// Globals
__global__ void GenerateSolverStates(const ATOMIC_SolverState* d_SolverStates, int StatesSize, ATOMIC_SolverState* d_NewSolverStates)
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t < StatesSize)
    {
        for (int i = 0; i < 4; i++)
        {
            ATOMIC_SolverState Candidate = d_SolverStates[t];
            int2 CurrentMoveStep = ATOMIC_Steps::GetStepByIndex(i);

            bool bMoveValid = Candidate.SceneState.MovePlayer(CurrentMoveStep);
            Candidate.SceneState.UpdatePhysics();
            Candidate.StepState.AddStep(CurrentMoveStep);
            Candidate.WinState = Candidate.SceneState.bIsWin();
            Candidate.ValidState = bMoveValid;
            d_NewSolverStates[i + 4 * t] = Candidate;
        }
    }
}
//Todo optimize this
__global__ void MarkInvalidDuplicatesFromGlobal(ATOMIC_SolverState* StatesToMark, int N_StatesToMark, ATOMIC_SolverState* StatesFind, int N_StatesFind)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_StatesToMark)
    {
        return;
    }
    for (int i = 0; i < N_StatesFind; i++)
    {
        if (StatesToMark[idx].SceneState == StatesFind[i].SceneState)
        {
            StatesToMark[idx].ValidState = false;
            break;
        }
    }
}

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