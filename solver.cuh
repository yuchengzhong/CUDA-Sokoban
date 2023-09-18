#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <map>
#include <set>
#include "scene.h"
#include "vectorize_scene.cuh"
#include "math.cuh"
const int N = 1024;

__global__ void RenderScene_UV(const RenderData* SceneData, int NumData, float3* Out_Buffer, int2 OutBufferSize, float Time);
__global__ void RenderScene(const RenderData* SceneData, int NumData, int3 SceneSize, float3* Out_Buffer, int2 OutBufferSize, float Time);

vector<float3> Launch_RenderScene(const vector<RenderData>& SceneData, int3 SceneSize, int2 BufferSize, float Time);



//BFS
/*
States[]
while States not empty do:
    NewState[] # len(NewState) = 4.0 * len(States)
    NewStateValid[] # len(NewStateValid) = 4.0 * len(States)
    do in every thread t:
        for i, CurrentMoveStep in AllMovement: # AllMovement: up down left right
            Valid, CandidateState = States[t].Move(CurrentMoveStep)
            NewState[i + 4*t] = CandidateState
            NewStateValid[i + 4*t] = Valid
    NewStateValid <- scan: NewStateValid[Valid]
    States <- remove_duplicate: NewStateValid

*/
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
    template<typename T>
    static bool Find(const vector<T>& Array, const T& ItemToFind)
    {
        for (int i=0;i< Array.size();i++)
        {
            if (Array[i] %= ItemToFind)
            {
                return true;
            }
        }
        return false;
    }

    static vector<ATOMIC_Steps> Solve(const ATOMIC_Scene& InitialScene)
	{
        vector<ATOMIC_Scene> AllStates;
        vector<ATOMIC_Steps> AllSteps;
        vector<bool> AllWins;

		vector<ATOMIC_Scene> States;
		vector<ATOMIC_Steps> Steps;
        States.push_back(InitialScene);
        Steps.push_back(ATOMIC_Steps::GetEmptyStep());
        int IterIndex = 0;
        while (States.size() > 0)
        {
            cout << "IterIndex:" << IterIndex << "\n";
            vector<ATOMIC_Scene> NewStates(States.size() * 4);
            vector<ATOMIC_Steps> NewSteps(States.size() * 4);
            vector<bool> NewWins(States.size() * 4);
            vector<bool> ValidStates(States.size() * 4);
            for (int t = 0; t < States.size(); t++) //Thread
            {
                for (int i = 0; i < 4; i++)
                {
                    ATOMIC_Scene Candidate = States[t];
                    int2 CurrentMoveStep = ATOMIC_Steps::GetStepByIndex(i);
                    //cout << "CurrentMoveStep:" << CurrentMoveStep.x << "," << CurrentMoveStep.y << "\n";
                    bool bMoveValid = Candidate.MovePlayer(CurrentMoveStep);
                    Candidate.UpdatePhysics();
                    bool bWin = Candidate.bIsWin();

                    //cout << "bMoveValid:" << bMoveValid << "\n";
                    ATOMIC_Steps CurrentSteps = Steps[t];
                    CurrentSteps.AddStep(CurrentMoveStep);

                    ValidStates[i + 4 * t] = bMoveValid;
                    NewWins[i + 4 * t] = bWin;
                    NewStates[i + 4 * t] = Candidate;
                    NewSteps[i + 4 * t] = CurrentSteps;
                }
            }
            //Scan
            vector<ATOMIC_Scene> DuplicatedStates = Scan(NewStates, ValidStates);
            vector<ATOMIC_Steps> DuplicatedSteps = Scan(NewSteps, ValidStates);
            vector<bool> DuplicatedWins = Scan(NewWins, ValidStates);
            cout << "DuplicatedStates:" << DuplicatedStates.size() << "\n";
            //Remove Duplicate
            vector<ATOMIC_Scene> NonDuplicatedStates;
            vector<ATOMIC_Steps> NonDuplicatedSteps;
            vector<bool> NonDuplicatedWins;
            for (int i = 0; i < DuplicatedStates.size(); i++)
            {
                if (!Find(AllStates, DuplicatedStates[i]))
                {
                    NonDuplicatedStates.push_back(DuplicatedStates[i]);
                    NonDuplicatedSteps.push_back(DuplicatedSteps[i]);
                    NonDuplicatedWins.push_back(DuplicatedWins[i]);
                    AllStates.push_back(DuplicatedStates[i]);
                    AllSteps.push_back(DuplicatedSteps[i]);
                    AllWins.push_back(DuplicatedWins[i]);
                }
            }
            cout << "NonDuplicatedStates:" << NonDuplicatedStates.size() << "\n";
            States = NonDuplicatedStates;
            Steps = NonDuplicatedSteps;
            IterIndex++;
            cout << "\n";
            cout << "\n";
        }
        cout << "AllSteps:" << AllSteps.size() << "\n";

        vector<ATOMIC_Steps> WinSteps = Scan(AllSteps, AllWins);
        return WinSteps;
	}
};