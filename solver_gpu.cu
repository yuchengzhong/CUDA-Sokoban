#include "solver_gpu.cuh"
#include "solver_gpu_base.cuh"

/*
    static vector<ATOMIC_Steps> Solve(const ATOMIC_Scene& InitialScene, bool ShortestOnly)
    {
        vector<ATOMIC_SolverState> AllSolverStates;

        vector<ATOMIC_SolverState> SolverStates;
        SolverStates.push_back(ATOMIC_SolverState{ InitialScene, ATOMIC_Steps::GetEmptyStep(), false });
        int IterIndex = 0;
        while (SolverStates.size() > 0)
        {
            vector<ATOMIC_SolverState> NewSolverStates(SolverStates.size() * 4);
            for (int t = 0; t < SolverStates.size(); t++) //Thread
            {
                for (int i = 0; i < 4; i++)
                {
                    ATOMIC_SolverState Candidate = SolverStates[t];
                    int2 CurrentMoveStep = ATOMIC_Steps::GetStepByIndex(i);
                    bool bMoveValid = Candidate.SceneState.MovePlayer(CurrentMoveStep);
                    Candidate.SceneState.UpdatePhysics();

                    Candidate.StepState.AddStep(CurrentMoveStep);

                    Candidate.WinState = Candidate.SceneState.bIsWin();
                    Candidate.ValidState = bMoveValid;
                    NewSolverStates[i + 4 * t] = Candidate;
                }
            }
            //Scan
            vector<ATOMIC_SolverState> DuplicatedSolverStates = Scan(NewSolverStates);
            //Remove Duplicate SolverState which has same ATOMIC_Scene
            vector<ATOMIC_SolverState> NonDuplicatedSolverStates;
            for (int i = 0; i < DuplicatedSolverStates.size(); i++)
            {
                if (!Find(AllSolverStates, DuplicatedSolverStates[i].SceneState))
                {
                    NonDuplicatedSolverStates.push_back(DuplicatedSolverStates[i]);
                    AllSolverStates.push_back(DuplicatedSolverStates[i]);
                }
            }
            SolverStates = NonDuplicatedSolverStates;
            IterIndex++;
            if (ShortestOnly && Any(SolverStates))
            {
                break;
            }
        }
        vector<ATOMIC_Steps> WinSteps = Scan_SolverState(AllSolverStates);
        return WinSteps;
    }
*/
vector<ATOMIC_Steps> GPU_Solver::Solve(const ATOMIC_Scene& InitialScene, bool ShortestOnly, bool Debug)
{
    cudaError_t CudaError = cudaGetLastError();
    //
    const unsigned int N_ThreadsPerBlock = 32;
    thrust::device_vector<ATOMIC_SolverState> AllSolverStates;
    thrust::device_vector<ATOMIC_SolverState> WinSolverStates;
    //
    thrust::device_vector<ATOMIC_SolverState> SolverStates(1);
    SolverStates[0] = ATOMIC_SolverState{ InitialScene, ATOMIC_Steps::GetEmptyStep(), false };
    int IterIndex = 0;
    while (SolverStates.size() > 0)
    {
        size_t StatesSize = SolverStates.size();
        thrust::device_vector<ATOMIC_SolverState> NewSolverStates;
        NewSolverStates.resize(StatesSize * 4);
        unsigned int N_Blocks = static_cast<unsigned int>((StatesSize + N_ThreadsPerBlock - 1) / N_ThreadsPerBlock);
        GenerateSolverStates << <N_Blocks, N_ThreadsPerBlock >> > (thrust::raw_pointer_cast(SolverStates.data()), static_cast<int>(StatesSize), thrust::raw_pointer_cast(NewSolverStates.data()));
        cudaDeviceSynchronize();
        if (Debug)
        {
            CudaError = cudaGetLastError();
            if (CudaError != cudaSuccess)
            {
                printf("CUDA Error GenerateSolverStates: %s\n", cudaGetErrorString(CudaError));
            }
        }
        // Scan
        thrust::device_vector<ATOMIC_SolverState> DuplicatedSolverStates = Scan(NewSolverStates, IsSolverStateValid());
        // TODO: Remove duplicated in DuplicatedSolverStates
        
        // Same, but with global culling
        N_Blocks = static_cast<unsigned int>((DuplicatedSolverStates.size() + N_ThreadsPerBlock - 1) / N_ThreadsPerBlock);
        MarkInvalidDuplicatesFromGlobal << <N_Blocks, N_ThreadsPerBlock >> > (thrust::raw_pointer_cast(DuplicatedSolverStates.data()), DuplicatedSolverStates.size(), thrust::raw_pointer_cast(AllSolverStates.data()), AllSolverStates.size());
        cudaDeviceSynchronize();
        if (Debug)
        {
            CudaError = cudaGetLastError();
            if (CudaError != cudaSuccess)
            {
                printf("CUDA Error MarkInvalidDuplicatesFromGlobal: %s\n", cudaGetErrorString(CudaError));
            }
        }
        SolverStates = Scan(DuplicatedSolverStates, IsSolverStateValid());
        // Add to AllSolverStates
        AllSolverStates.insert(AllSolverStates.end(), SolverStates.begin(), SolverStates.end());
        //Scan win state
        WinSolverStates = Scan(SolverStates, IsSolverStateWin());
        if (ShortestOnly && WinSolverStates.size() > 0)
        {
            break;
        }
        IterIndex++;
    }
    thrust::device_vector<ATOMIC_Steps> WinSteps = ExtractSteps(WinSolverStates);
    vector<ATOMIC_Steps> CPU_WinSteps(WinSteps.begin(), WinSteps.end());
    return CPU_WinSteps;
}