#include "solver_gpu_global.cuh"

// Globals
__global__ void GenerateSolverStates(const ATOMIC_SolverState* SolverStates, const STATIC_SceneBlock* SceneBlock, const int N_SolverStates, ATOMIC_SolverState* d_NewSolverStates)
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

        bool bMoveValid = Candidate.SceneState.MovePlayer(CurrentMoveStep, SceneBlock[Candidate.SceneState.SceneIndex]);
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
    __shared__ struct 
    {
        unsigned char ActorState;
        uchar3 Location;
        unsigned char Rotation;
    } SHARED_ActorsFind[CHUNK_SIZE * ATOMIC_MAX_ACTORS];

    __shared__ int SHARED_SceneIndex[CHUNK_SIZE];
    __shared__ int SHARED_ActorCount[CHUNK_SIZE];

    if (N_StatesFind <= 0 || idx >= N_StatesToMark)
    {
        return;
    }

    int N_Chunks = (N_StatesFind + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const ATOMIC_SolverState CurrentStateToMark = StatesToMark[idx];
    for (int Chunk = 0; Chunk < N_Chunks; Chunk++)
    {
        int ChunkIdx = Chunk * CHUNK_SIZE + threadIdx.x;
        if (ChunkIdx < N_StatesFind)
        {
            const ATOMIC_SolverState CurrentStateToFind = StatesFind[ChunkIdx];
            SHARED_ActorCount[threadIdx.x] = CurrentStateToFind.SceneState.ActorCount;
            for (int j = 0; j < SHARED_ActorCount[threadIdx.x]; ++j)
            {
                int Index = threadIdx.x * ATOMIC_MAX_ACTORS + j;
                SHARED_ActorsFind[Index].ActorState = { CurrentStateToFind.SceneState.Actors[j].ActorState };
                SHARED_ActorsFind[Index].Location = CurrentStateToFind.SceneState.Actors[j].Location;
                SHARED_ActorsFind[Index].Rotation = CurrentStateToFind.SceneState.Actors[j].Rotation;
            }
            SHARED_SceneIndex[threadIdx.x] = CurrentStateToFind.SceneState.SceneIndex;
        }
        __syncthreads();
        for (int i = 0; i < CHUNK_SIZE && (Chunk * CHUNK_SIZE + i) < N_StatesFind; i++)
        {
            if (SHARED_SceneIndex[i] != CurrentStateToMark.SceneState.SceneIndex ||
                SHARED_ActorCount[i] != CurrentStateToMark.SceneState.ActorCount)
            {
                continue;
            }
            bool bCanSkip = false;
            for (int j = 0; j < CurrentStateToMark.SceneState.ActorCount; ++j)
            {
                int Index = i * ATOMIC_MAX_ACTORS + j;
                if (CurrentStateToMark.SceneState.Actors[j].ActorState != SHARED_ActorsFind[Index].ActorState ||
                    CurrentStateToMark.SceneState.Actors[j].Location != SHARED_ActorsFind[Index].Location ||
                    CurrentStateToMark.SceneState.Actors[j].Rotation != SHARED_ActorsFind[Index].Rotation
                    )
                {
                    bCanSkip = true;
                    break;
                }
            }
            if (!bCanSkip)
            {
                StatesToMark[idx].ValidState = false;
                break;
            }
        }
        __syncthreads();
    }
}
#undef CHUNK_SIZE
