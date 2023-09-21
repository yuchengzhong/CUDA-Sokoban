#include "solver_gpu.cuh"
#include "solver_gpu_base.cuh"
#include "solver_gpu_global.cuh"

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

#define GPU_SOLVER_STATE_FIRST_ALLOC 2048
vector<ATOMIC_Steps> GPU_Solver::Solve(const ATOMIC_Scene& InitialScene, const STATIC_SceneBlock& SceneBlock, bool ShortestOnly, bool Debug)
{
    Timer SolverTimer;
    double T_NewSolverStates_resize = 0.0;
    double T_GenerateSolverStates = 0.0;
    double T_Scan_NewSolverStates = 0.0;
    double T_MarkInvalidDuplicatesFromGlobal = 0.0;
    double T_Scan_DuplicatedSolverStates = 0.0;
    double T_insert = 0.0;
    double T_Scan_SolverStates = 0.0;
    double T_ExtractSteps = 0.0;
    //
    cudaError_t CudaError = cudaGetLastError();
    //
    const unsigned int N_ThreadsPerBlock = 32;
    thrust::device_vector<STATIC_SceneBlock> STATIC_SceneBlocks(1);
    STATIC_SceneBlocks[0] = SceneBlock;

    thrust::device_vector<ATOMIC_SolverState> AllSolverStates;
    //
    thrust::device_vector<ATOMIC_SolverState> SolverStates(GPU_SOLVER_STATE_FIRST_ALLOC);
    thrust::device_vector<ATOMIC_SolverState> NewSolverStates(GPU_SOLVER_STATE_FIRST_ALLOC * 4); // TODO: Ping pong

    SolverStates[0] = ATOMIC_SolverState{ InitialScene, ATOMIC_Steps::GetEmptyStep(), false };
    size_t N_SolverStates = 1;
    size_t N_NewSolverStates = 0;
    int IterIndex = 0;

    auto PredValid = IsSolverStateValid();
    auto PredWin = IsSolverStateWin();
    while (N_SolverStates > 0)
    {
        SolverTimer.Start();
        N_NewSolverStates = N_SolverStates * 4;
        cout << "First N_NewSolverStates:" << N_NewSolverStates << "\n";
        if (NewSolverStates.size() < N_NewSolverStates)
        {
            NewSolverStates = thrust::device_vector<ATOMIC_SolverState>(N_NewSolverStates);
        }
        T_NewSolverStates_resize += SolverTimer.Reset(string("NewSolverStates.resize"), false);

        SolverTimer.Start();
        unsigned int N_Blocks = static_cast<unsigned int>((N_SolverStates + N_ThreadsPerBlock - 1) / N_ThreadsPerBlock);
        GenerateSolverStates << <N_Blocks, N_ThreadsPerBlock >> > (thrust::raw_pointer_cast(SolverStates.data()), thrust::raw_pointer_cast(STATIC_SceneBlocks.data()), static_cast<int>(N_SolverStates), thrust::raw_pointer_cast(NewSolverStates.data()));
        cudaDeviceSynchronize();
        T_GenerateSolverStates += SolverTimer.Reset(string("GenerateSolverStates"), false);
        if (Debug)
        {
            CudaError = cudaGetLastError();
            if (CudaError != cudaSuccess)
            {
                printf("CUDA Error GenerateSolverStates: %s\n", cudaGetErrorString(CudaError));
            }
        }
        // Scan
        SolverTimer.Start();
        cout << "T_Scan_NewSolverStates N_NewSolverStates:" << N_NewSolverStates << "\n";
        N_NewSolverStates = Scan(NewSolverStates, N_NewSolverStates, PredValid);
        T_Scan_NewSolverStates += SolverTimer.Reset(string("Scan NewSolverStates"), false);
        // TODO: Remove duplicated in DuplicatedSolverStates
        
        // Same, but with global culling
        SolverTimer.Start();
        N_Blocks = static_cast<unsigned int>((N_NewSolverStates + N_ThreadsPerBlock - 1) / N_ThreadsPerBlock);
        MarkInvalidDuplicatesFromGlobal << <N_Blocks, N_ThreadsPerBlock >> > (thrust::raw_pointer_cast(NewSolverStates.data()), static_cast<int>(N_NewSolverStates), thrust::raw_pointer_cast(AllSolverStates.data()), static_cast<int>(AllSolverStates.size()));
        cudaDeviceSynchronize();
        T_MarkInvalidDuplicatesFromGlobal += SolverTimer.Reset(string("MarkInvalidDuplicatesFromGlobal"), false);
        if (Debug)
        {
            CudaError = cudaGetLastError();
            if (CudaError != cudaSuccess)
            {
                printf("CUDA Error MarkInvalidDuplicatesFromGlobal: %s\n", cudaGetErrorString(CudaError));
            }
        }
        SolverTimer.Start();
        N_NewSolverStates = Scan(NewSolverStates, N_NewSolverStates, PredValid);
        cout << "T_Scan_DuplicatedSolverStates N_NewSolverStates:" << N_NewSolverStates << "\n";
        if (SolverStates.size() < N_NewSolverStates) // If pingpong, replace this
        {
            SolverStates = thrust::device_vector<ATOMIC_SolverState>(N_NewSolverStates);
        }
        thrust::copy(NewSolverStates.begin(), NewSolverStates.begin() + N_NewSolverStates, SolverStates.begin()); // first, last, result
        N_SolverStates = N_NewSolverStates;
        T_Scan_DuplicatedSolverStates += SolverTimer.Reset(string("Scan DuplicatedSolverStates"), false);
        // Add to AllSolverStates
        SolverTimer.Start();
        AllSolverStates.insert(AllSolverStates.end(), SolverStates.begin(), SolverStates.end());
        T_insert += SolverTimer.Reset(string("insert"), false);
        //Scan win state
        SolverTimer.Start();
        N_NewSolverStates = Scan(NewSolverStates, N_NewSolverStates, PredWin);
        T_Scan_SolverStates += SolverTimer.Reset(string("Scan SolverStates"), false);
        if (ShortestOnly && N_NewSolverStates > 0)
        {
            break;
        }
        IterIndex++;
    }
    cout << "GPU iters: " << IterIndex << "\n";
    SolverTimer.Start();
    thrust::device_vector<ATOMIC_Steps> WinSteps = ExtractSteps(NewSolverStates, N_NewSolverStates);
    T_ExtractSteps += SolverTimer.Reset(string("ExtractSteps"), false);
    vector<ATOMIC_Steps> CPU_WinSteps(WinSteps.begin(), WinSteps.end());


    cout << "\nDetails: " << "\n";
    cout << "NewSolverStates_resize " << T_NewSolverStates_resize << "ms\n";
    cout << "GenerateSolverStates " << T_GenerateSolverStates << "ms\n";
    cout << "Scan_NewSolverStates " << T_Scan_NewSolverStates << "ms\n";
    cout << "MarkInvalidDuplicatesFromGlobal " << T_MarkInvalidDuplicatesFromGlobal << "ms\n";
    cout << "Scan_DuplicatedSolverStates " << T_Scan_DuplicatedSolverStates << "ms\n";
    cout << "insert " << T_insert << "ms\n";
    cout << "Scan_SolverStates " << T_Scan_SolverStates << "ms\n";
    cout << "ExtractSteps " << T_ExtractSteps << "ms\n";
    cout << "All " << (T_NewSolverStates_resize + T_GenerateSolverStates + T_Scan_NewSolverStates + T_MarkInvalidDuplicatesFromGlobal + T_Scan_DuplicatedSolverStates + T_insert + T_Scan_SolverStates + T_ExtractSteps) << "ms\n";
    cout << "\n";

    return CPU_WinSteps;
}