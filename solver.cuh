#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <map>
#include <set>
#include "scene.h"
#include "vectorize_scene.cuh"
#include "math.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
const int N = 1024;

__global__ void RenderScene_UV(const RenderData* SceneData, int NumData, float3* Out_Buffer, int2 OutBufferSize, float Time);
__global__ void RenderScene(const RenderData* SceneData, int NumData, int3 SceneSize, float3* Out_Buffer, int2 OutBufferSize, float Time);

vector<float3> Launch_RenderScene(const vector<RenderData>& SceneData, int3 SceneSize, int2 BufferSize, float Time);

struct ATOMIC_SolverState
{
    ATOMIC_Scene SceneState;
    ATOMIC_Steps StepState;
    bool WinState = false;
    bool ValidState = false;
};
struct CPU_Solver
{
public:
    template<typename T>
    static vector<T> Scan(const vector<T> ScanArray, const vector<bool> ValidArray)
    {
        vector<T> Result;
        for (int i=0;i< ScanArray.size();i++)
        {
            if (ValidArray[i])
            {
                Result.push_back(ScanArray[i]);
            }
        }
        return Result;
    }
    static vector<ATOMIC_SolverState> Scan(const vector<ATOMIC_SolverState> ScanArray)
    {
        vector<ATOMIC_SolverState> Result;
        for (int i=0;i< ScanArray.size();i++)
        {
            if (ScanArray[i].ValidState)
            {
                Result.push_back(ScanArray[i]);
            }
        }
        return Result;
    }
    static bool Any(const vector<bool>& Array)
    {
        for (int i=0;i< Array.size();i++)
        {
            if (Array[i])
            {
                return true;
            }
        }
        return false;
    }
    static bool Find(const vector<ATOMIC_SolverState>& Array, const ATOMIC_Scene& ItemToFind)
    {
        for (int i = 0; i < Array.size(); i++)
        {
            if (Array[i].SceneState %= ItemToFind)
            {
                return true;
            }
        }
        return false;
    }
    static bool Any(const vector<ATOMIC_SolverState>& Array)
    {
        for (int i=0;i< Array.size();i++)
        {
            if (Array[i].WinState)
            {
                return true;
            }
        }
        return false;
    }
    static vector<ATOMIC_Steps> Scan_SolverState(const vector<ATOMIC_SolverState> ScanArray)
    {
        vector<ATOMIC_Steps> Result;
        for (int i = 0; i < ScanArray.size(); i++)
        {
            if (ScanArray[i].WinState)
            {
                Result.push_back(ScanArray[i].StepState);
            }
        }
        return Result;
    }
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
            //cout << DuplicatedSolverStates.size() << "\n";
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
};


__global__ void GenerateSolverStates(const ATOMIC_SolverState* d_SolverStates, int StatesSize, ATOMIC_SolverState* d_NewSolverStates);

thrust::device_vector<ATOMIC_SolverState> Scan(const thrust::device_vector<ATOMIC_SolverState>& NewSolverStates);
struct GPU_Solver
{
public:
    static vector<ATOMIC_Steps> Solve(const ATOMIC_Scene& InitialScene, bool ShortestOnly);
};