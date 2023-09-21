#include "solver_gpu_global.cuh"

// Globals
__global__ void GenerateSolverStates(const ATOMIC_SolverState* SolverStates, const int N_SolverStates, ATOMIC_SolverState* d_NewSolverStates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_SolverStates)
    {
        return;
    }
    const int2 Steps[4] = { { 1,0 } ,{ -1,0 } ,{ 0,1 } , { 0,-1 } };
    for (int i = 0; i < 4; i++)
    {
        ATOMIC_SolverState Candidate = SolverStates[idx];
        int2 CurrentMoveStep = Steps[i];// ATOMIC_Steps::GetStepByIndex(i);

        bool bMoveValid = Candidate.SceneState.MovePlayer(CurrentMoveStep);
        Candidate.SceneState.UpdatePhysics();
        Candidate.StepState.AddStep(CurrentMoveStep);
        Candidate.WinState = Candidate.SceneState.bIsWin();
        Candidate.ValidState = bMoveValid;
        d_NewSolverStates[i + 4 * idx] = Candidate;
    }
}
#define CHUNK_SIZE 32
__global__ void MarkInvalidDuplicatesFromGlobal(ATOMIC_SolverState* StatesToMark, const int N_StatesToMark, const ATOMIC_SolverState* StatesFind, const int N_StatesFind)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ Actor SHARED_ActorsFind[CHUNK_SIZE * ATOMIC_MAX_ACTORS];
    __shared__ int SHARED_ActorCount;


    if (threadIdx.x == 0)
    {
        SHARED_ActorCount = StatesToMark[0].SceneState.ActorCount;
    }
    __syncthreads();

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
            for (int j = 0; j < SHARED_ActorCount; ++j)
            {
                SHARED_ActorsFind[threadIdx.x * SHARED_ActorCount + j] = StatesFind[ChunkIdx].SceneState.Actors[j];
            }
        }
        __syncthreads();
        for (int i = 0; i < CHUNK_SIZE && (Chunk * CHUNK_SIZE + i) < N_StatesFind; i++)
        {
            bool bCanSkip = false;
            for (int j = 0; j < SHARED_ActorCount; ++j)
            {
                if (StatesToMark[idx].SceneState.Actors[j] != SHARED_ActorsFind[i * SHARED_ActorCount + j])
                {
                    bCanSkip = true;
                    break;
                }
            }
            if (bCanSkip)
            {
                break;
            }
            StatesToMark[idx].ValidState = false;
        }
        __syncthreads();
    }
}
#undef CHUNK_SIZE
