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
#define CHUNK_SIZE 32
__global__ void MarkInvalidDuplicatesFromGlobal(ATOMIC_SolverState* StatesToMark, int N_StatesToMark, ATOMIC_SolverState* StatesFind, int N_StatesFind)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ Actor SHARED_ActorsFind[CHUNK_SIZE * ATOMIC_MAX_ACTORS];
    __shared__ int SHARED_ActorCount[CHUNK_SIZE];

    if (idx >= N_StatesToMark)
    {
        return;
    }
    int N_Chunks = (N_StatesFind + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int Chunk = 0; Chunk < N_Chunks; Chunk++)
    {
        int ChunkIdx = Chunk * CHUNK_SIZE + threadIdx.x;
        if (ChunkIdx < N_StatesFind)
        {
            for (int j = 0; j < ATOMIC_MAX_ACTORS; ++j)
            {
                SHARED_ActorsFind[threadIdx.x * ATOMIC_MAX_ACTORS + j] = StatesFind[ChunkIdx].SceneState.Actors[j];
            }
            SHARED_ActorCount[threadIdx.x] = StatesFind[ChunkIdx].SceneState.ActorCount;
        }
        __syncthreads();
        for (int i = 0; i < CHUNK_SIZE && (Chunk * CHUNK_SIZE + i) < N_StatesFind; i++)
        {
            bool bCanSkip = false;
            if (StatesToMark[idx].SceneState.ActorCount == SHARED_ActorCount[i])
            {
                for (int j = 0; j < ATOMIC_MAX_ACTORS; ++j)
                {
                    if (StatesToMark[idx].SceneState.Actors[j] != SHARED_ActorsFind[i * ATOMIC_MAX_ACTORS + j])
                    {
                        bCanSkip = true;
                        StatesToMark[idx].ValidState = false;
                        break;
                    }
                }
            }
            if (bCanSkip)
            {
                break;
            }
        }
        __syncthreads();
    }
}
#undef CHUNK_SIZE

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