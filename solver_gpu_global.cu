#include "solver_gpu_global.cuh"

// Globals
__global__ void GenerateSolverStates(const ATOMIC_SolverState* SolverStates, const STATIC_SceneBlock* SceneBlock, const uint32_t N_SolverStates, ATOMIC_SolverState* d_NewSolverStates)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_SolverStates)
    {
        return;
    }
    const int2 Steps[4] = { { 1,0 } ,{ -1,0 } ,{ 0,1 } , { 0,-1 } };
    for (uint32_t i = 0; i < 4; i++)
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
__global__ void MarkInvalidDuplicatesFromGlobal(ATOMIC_SolverState* StatesToMark, const uint32_t N_StatesToMark, const ATOMIC_SolverState* StatesFind, const uint32_t N_StatesFind)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ uint32_t SHARED_ActorsFind[CHUNK_SIZE * ATOMIC_MAX_ACTORS];
    __shared__ unsigned char SHARED_SceneIndex[CHUNK_SIZE];
    __shared__ unsigned char SHARED_ActorCount[CHUNK_SIZE];

    if (N_StatesFind <= 0 || idx >= N_StatesToMark)
    {
        return;
    }

    uint32_t N_Chunks = (N_StatesFind + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const ATOMIC_Scene CurrentStateToMark = StatesToMark[idx].SceneState;
    for (uint32_t Chunk = 0; Chunk < N_Chunks; Chunk++)
    {
        uint32_t ChunkIdx = Chunk * CHUNK_SIZE + threadIdx.x;
        if (ChunkIdx < N_StatesFind)
        {
            const ATOMIC_Scene CurrentStateToFind = StatesFind[ChunkIdx].SceneState;
            SHARED_ActorCount[threadIdx.x] = CurrentStateToFind.ActorCount;
            SHARED_SceneIndex[threadIdx.x] = CurrentStateToFind.SceneIndex;
            for (uint32_t j = 0; j < SHARED_ActorCount[threadIdx.x]; ++j)
            {
                uint32_t Index = threadIdx.x * ATOMIC_MAX_ACTORS + j;
                SHARED_ActorsFind[Index] = Pack1UChar1UChar3ToUINT32(CurrentStateToFind.Actors[j].ActorState, CurrentStateToFind.Actors[j].Location);
            }
        }
        __syncthreads();
        for (uint32_t i = 0; i < CHUNK_SIZE && (Chunk * CHUNK_SIZE + i) < N_StatesFind; i++)
        {
            if (SHARED_SceneIndex[i] != CurrentStateToMark.SceneIndex ||
                SHARED_ActorCount[i] != CurrentStateToMark.ActorCount)
            {
                continue;
            }
            bool bCanSkip = false;
            for (uint32_t j = 0; j < CurrentStateToMark.ActorCount; ++j)
            {
                uint32_t Index = i * ATOMIC_MAX_ACTORS + j;
                uint32_t Packed = Pack1UChar1UChar3ToUINT32(CurrentStateToMark.Actors[j].ActorState, CurrentStateToMark.Actors[j].Location);
                if (Packed != SHARED_ActorsFind[Index])
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
