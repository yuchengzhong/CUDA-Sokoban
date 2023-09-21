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
vector<ATOMIC_Steps> GPU_Solver::Solve(const ATOMIC_Scene& InitialScene, bool ShortestOnly, bool Debug)
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
    thrust::device_vector<ATOMIC_SolverState> AllSolverStates;
    thrust::device_vector<ATOMIC_SolverState> WinSolverStates;
    //
    thrust::device_vector<ATOMIC_SolverState> SolverStates(1);
    SolverStates[0] = ATOMIC_SolverState{ InitialScene, ATOMIC_Steps::GetEmptyStep(), false };
    int IterIndex = 0;
    while (SolverStates.size() > 0)
    {
        size_t StatesSize = SolverStates.size();

        SolverTimer.Start();
        thrust::device_vector<ATOMIC_SolverState> NewSolverStates;
        NewSolverStates.resize(StatesSize * 4);
        T_NewSolverStates_resize += SolverTimer.Reset(string("NewSolverStates.resize"), false);

        SolverTimer.Start();
        unsigned int N_Blocks = static_cast<unsigned int>((StatesSize + N_ThreadsPerBlock - 1) / N_ThreadsPerBlock);
        GenerateSolverStates << <N_Blocks, N_ThreadsPerBlock >> > (thrust::raw_pointer_cast(SolverStates.data()), static_cast<int>(StatesSize), thrust::raw_pointer_cast(NewSolverStates.data()));
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
        thrust::device_vector<ATOMIC_SolverState> DuplicatedSolverStates = Scan(NewSolverStates, IsSolverStateValid());
        T_Scan_NewSolverStates += SolverTimer.Reset(string("Scan NewSolverStates"), false);
        // TODO: Remove duplicated in DuplicatedSolverStates
        
        // Same, but with global culling
        SolverTimer.Start();
        N_Blocks = static_cast<unsigned int>((DuplicatedSolverStates.size() + N_ThreadsPerBlock - 1) / N_ThreadsPerBlock);
        MarkInvalidDuplicatesFromGlobal << <N_Blocks, N_ThreadsPerBlock >> > (thrust::raw_pointer_cast(DuplicatedSolverStates.data()), static_cast<int>(DuplicatedSolverStates.size()), thrust::raw_pointer_cast(AllSolverStates.data()), static_cast<int>(AllSolverStates.size()));
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
        SolverStates = Scan(DuplicatedSolverStates, IsSolverStateValid());
        T_Scan_DuplicatedSolverStates += SolverTimer.Reset(string("Scan DuplicatedSolverStates"), false);
        // Add to AllSolverStates
        SolverTimer.Start();
        AllSolverStates.insert(AllSolverStates.end(), SolverStates.begin(), SolverStates.end());
        T_insert += SolverTimer.Reset(string("insert"), false);
        //Scan win state
        SolverTimer.Start();
        WinSolverStates = Scan(SolverStates, IsSolverStateWin());
        T_Scan_SolverStates += SolverTimer.Reset(string("Scan SolverStates"), false);
        if (ShortestOnly && WinSolverStates.size() > 0)
        {
            break;
        }
        IterIndex++;
    }
    cout << "GPU iters: " << IterIndex << "\n";
    SolverTimer.Start();
    thrust::device_vector<ATOMIC_Steps> WinSteps = ExtractSteps(WinSolverStates);
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
    cout << "\n";

    return CPU_WinSteps;
}