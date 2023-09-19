#pragma once
#include "scene.h"
#include "math.cuh"
#include <string>

#define ATOMIC_MAX_BLOCKS 16*16*16
#define ATOMIC_MAX_ACTORS 64
#define ATOMIC_MAX_STEP 256


struct ATOMIC_Steps
{
public:
    int2 Step[ATOMIC_MAX_STEP];
    int StepCount = 0;
    __host__ __device__ bool AddStep(int2 Position)
    {
        if (StepCount < ATOMIC_MAX_STEP)
        {
            Step[StepCount] = Position;
            StepCount++;
            return true;
        }
        return false;
    }
    __host__ __device__ static ATOMIC_Steps GetEmptyStep()
    {
        ATOMIC_Steps Result;
        Result.Step[0] = { 0,0 };
        return Result;
    }
    __host__ __device__ static int2 GetStepByIndex(int i)
    {
        int2 Step = { 0,0 };
        if (i == 0)
        {
            Step = { 1,0 };
        }
        else if (i == 1)
        {
            Step = { -1,0 };
        }
        else if (i == 2)
        {
            Step = { 0,1 };
        }
        else if (i == 3)
        {
            Step = { 0,-1 };
        }
        return Step;
    }
};

struct ATOMIC_Scene
{
public:
    uint8_t SceneBlock[ATOMIC_MAX_BLOCKS];
    Actor Actors[ATOMIC_MAX_ACTORS];
    int3 SceneSize = {0,0,0};
    int ActorCount = 0;

    __host__ bool InitialFromScene(const Scene& SourceScene)
    {
        if (SourceScene.SceneBlock.size() > ATOMIC_MAX_BLOCKS)
        {
            return false;
        }
        if (SourceScene.Actors.size() > ATOMIC_MAX_ACTORS)
        {
            return false;
        }
        for (size_t i = 0; i < SourceScene.SceneBlock.size(); ++i)
        {
            SceneBlock[i] = SourceScene.SceneBlock[i];
        }
        for (size_t i = 0; i < SourceScene.Actors.size(); ++i)
        {
            Actors[i] = SourceScene.Actors[i];
        }
        SceneSize = SourceScene.SceneSize;
        ActorCount = static_cast<int>(SourceScene.Actors.size());
        return true;
    }

    __host__ bool SetupScene(const vector<uint8_t>& SceneBlock_, const vector<Actor>& Actors_, int3 SceneSize_)
    {
        assert(SceneBlock_.size() == SceneSize_.x * SceneSize_.y * SceneSize_.z);
        if (SceneBlock_.size() > ATOMIC_MAX_BLOCKS)
        {
            return false;
        }
        if (Actors_.size() > ATOMIC_MAX_ACTORS)
        {
            return false;
        }
        SceneSize = SceneSize_;
        ActorCount = static_cast<int>(Actors_.size());
        for (size_t i = 0; i < SceneBlock_.size(); ++i)
        {
            SceneBlock[i] = SceneBlock_[i];
        }
        for (size_t i = 0; i < Actors_.size(); ++i)
        {
            Actors[i] = Actors_[i];
        }
        return true;
    }
    __host__ void Reset()
    {
        for (int i = 0; i < ATOMIC_MAX_BLOCKS; i++)
        {
            SceneBlock[i] = 0;
        }
        Actor defaultActor;
        for (int i = 0; i < ATOMIC_MAX_ACTORS; i++)
        {
            Actors[i] = defaultActor;
        }
        SceneSize = int3();
        ActorCount = 0;
    }

    __host__ __device__ bool bIsPositionOutOfBound(int3 Position) const
    {
        return Position.x < 0 || Position.x >= SceneSize.x || Position.y < 0 || Position.y >= SceneSize.y || Position.z < 0 || Position.z >= SceneSize.z;
    }
    __host__ __device__ bool MovePlayer(int2 Move)
    {
        if (Move.x == 0 && Move.y == 0)
        {
            return false;
        }
        if (abs(Move.x) == abs(Move.y) && abs(Move.x) == 1)
        {
            Move.y = 0;
        }
        for (int i = 0; i < ActorCount;i++)
        {
            Actor& CurrentActor = Actors[i];
            uint8_t ActorsType = CurrentActor.ActorType;
            if (ActorsType == SOKOBAN_PLAYER)
            {
                int3 NewPosition = { CurrentActor.Location.x + Move.x, CurrentActor.Location.y + Move.y, CurrentActor.Location.z };
                if (bIsPositionOutOfBound(NewPosition))
                {
                    return false;
                }
                int NewIndex = NewPosition.x + NewPosition.y * SceneSize.x + NewPosition.z * SceneSize.x * SceneSize.y;
                if (SceneBlock[NewIndex] == SOKOBAN_WALL)
                {
                    return false;
                }
                //Find box
                for (int j = 0; j < ActorCount; j++)
                {
                    Actor& ActorW = Actors[j];
                    if (ActorW.ActorType == SOKOBAN_BOX)
                    {
                        if (ActorW.Location == NewPosition) // if push
                        {
                            int BoxId = ActorW.Id;
                            int3 BoxNewPosition = { ActorW.Location.x + Move.x, ActorW.Location.y + Move.y, ActorW.Location.z };
                            if (bIsPositionOutOfBound(BoxNewPosition))
                            {
                                return false;
                            }
                            int BoxNewIndex = BoxNewPosition.x + BoxNewPosition.y * SceneSize.x + BoxNewPosition.z * SceneSize.x * SceneSize.y;
                            if (SceneBlock[BoxNewIndex] == SOKOBAN_WALL) // If box forward is wall
                            {
                                return false;
                            }
                            //Find box
                            for (int k = 0; k < ActorCount; k++)
                            {
                                const Actor& ActorW2 = Actors[k];
                                if (ActorW2.ActorType == SOKOBAN_BOX)
                                {
                                    if (ActorW2.Location == BoxNewPosition)
                                    {
                                        int Box2Id = ActorW2.Id;
                                        if (BoxId != Box2Id)
                                        {
                                            return false;
                                        }
                                    }
                                }
                            }
                            // If can push
                            ActorW.Location = BoxNewPosition;
                        }
                    }
                }
                CurrentActor.Location = NewPosition;
                return true;
            }
        }
        return false;
    }

    __host__ __device__ void UpdatePhysics()
    {
        //Do physics

        //Do after physics

        //Find box
        for (int i = 0; i < ActorCount; i++)
        {
            Actor& ActorW = Actors[i];
            if (ActorW.ActorType == SOKOBAN_BOX)
            {
                int3 BoxPosition = ActorW.Location;

                bool Continue = false;
                for (int j = 0; j < ActorCount; j++)
                {
                    Actor& ActorW2 = Actors[j];
                    if (ActorW2.ActorType == SOKOBAN_BOX_TARGET)
                    {
                        int3 TargetPosition = ActorW2.Location;
                        if (TargetPosition == BoxPosition)
                        {
                            ActorW.ActorState = SOKOBAN_ACTIVE;
                            Continue = true;
                            break;
                        }
                    }
                }
                if(Continue)
                {
                    continue;
                }
                else
                {
                    ActorW.ActorState = SOKOBAN_INACTIVE;
                }
            }
        }
    }
    __host__ __device__ bool bIsWin() const
    {
        for (int i = 0; i < ActorCount; i++)
        {
            const Actor& ActorW = Actors[i];
            if (ActorW.ActorType == SOKOBAN_BOX)
            {
                if (ActorW.ActorState == SOKOBAN_INACTIVE)
                {
                    return false;
                }
            }
        }
        return true;
    }
    vector<RenderData> GetRenderData() const
    {
        vector<RenderData> Result;
        for (int x = 0; x < SceneSize.x; x++)
        {
            for (int y = 0; y < SceneSize.y; y++)
            {
                for (int z = 0; z < SceneSize.z; z++)
                {
                    int Index = x + y * SceneSize.x + z * SceneSize.x * SceneSize.y;
                    uint8_t BlockType = SceneBlock[Index];
                    if (BlockType == SOKOBAN_WALL)
                    {
                        Result.push_back({ {x,y,z},SOKOBAN_WALL_COLOR , {1.0f,1.0f,1.0f}, {0.0f,0.0f,0.0f} });
                    }
                }
            }
        }
        for (int i = 0; i < ActorCount; i++)
        {
            const Actor& ActorW = Actors[i];
            int ActorsType = ActorW.ActorType;
            int ActorsState = ActorW.ActorState;
            int x = ActorW.Location.x;
            int y = ActorW.Location.y;
            int z = ActorW.Location.z;
            if (ActorsType == SOKOBAN_PLAYER_START)
            {
                Result.push_back({ {x,y,z},SOKOBAN_PLAYER_START_COLOR , {1.0f,0.125f,1.0f}, {0.0f,0.5f - 0.0625f,0.0f} });
            }
            else if (ActorsType == SOKOBAN_PLAYER)
            {
                Result.push_back({ {x,y,z},SOKOBAN_PLAYER_COLOR , {0.8f,0.8f,0.8f}, {0.0f,0.0f,0.0f} });
            }
            else if (ActorsType == SOKOBAN_BOX)
            {
                Result.push_back({ {x,y,z}, (ActorsState == 0 ? SOKOBAN_BOX_COLOR : SOKOBAN_BOX_COLOR_ACTIVE) , {0.95f,0.95f,0.95f}, {0.0f,0.025f,0.0f} });
            }
            else if (ActorsType == SOKOBAN_BOX_TARGET)
            {
                Result.push_back({ {x,y,z},SOKOBAN_BOX_TARGET_COLOR , {1.0f,0.125f,1.0f}, {0.0f,0.5f - 0.0625f,0.0f} });
            }
        }
        //printf("Atomic\n");
        return Result;
    }
    __host__ __device__ static bool bIsSame(const ATOMIC_Scene& SceneState1, const ATOMIC_Scene& SceneState2)
    {
        if (SceneState1.SceneSize != SceneState2.SceneSize)
        {
            return false;
        }
        for (int x = 0; x < SceneState1.SceneSize.x; x++)
        {
            for (int y = 0; y < SceneState1.SceneSize.y; y++)
            {
                for (int z = 0; z < SceneState1.SceneSize.z; z++)
                {
                    int Index = x + y * SceneState1.SceneSize.x + z * SceneState1.SceneSize.x * SceneState1.SceneSize.y;
                    if (SceneState1.SceneBlock[Index] != SceneState2.SceneBlock[Index])
                    {
                        return false;
                    }
                }
            }
        }
        if (SceneState1.ActorCount != SceneState2.ActorCount)
        {
            return false;
        }
        for (int i = 0; i < SceneState1.ActorCount; i++)
        {
            if (SceneState1.Actors[i] != SceneState2.Actors[i])
            {
                return false;
            }
        }
        return true;
    }
    void Debug()
    {
        for (int z = 0; z < SceneSize.z; z++)
        {
            for (int y = 0; y < SceneSize.y; y++)
            {
                for (int x = 0; x < SceneSize.x; x++)
                {
                    int Index = x + y * SceneSize.x + z * SceneSize.x * SceneSize.y;
                    printf("%d ", SceneBlock[Index]);
                }
                printf("\n");
            }
            printf("\n");
        }
        for (int i = 0; i < ActorCount; i++)
        {
            Actor CurrentActor = Actors[i];
            std::string ActorName = "Not Defined";
            if (CurrentActor.ActorType == SOKOBAN_PLAYER_START)
            {
                ActorName = "SOKOBAN_PLAYER_START";
            }
            else if (CurrentActor.ActorType == SOKOBAN_PLAYER)
            {
                ActorName = "SOKOBAN_PLAYER";
            }
            else if (CurrentActor.ActorType == SOKOBAN_BOX)
            {
                ActorName = "SOKOBAN_BOX";
            }
            else if (CurrentActor.ActorType == SOKOBAN_BOX_TARGET)
            {
                ActorName = "SOKOBAN_BOX_TARGET";
            }
            printf("%s:id-%d, state-%d, loc-(%d,%d,%d)\n", ActorName.c_str(), CurrentActor.Id, CurrentActor.ActorState, 
                CurrentActor.Location.x, CurrentActor.Location.y, CurrentActor.Location.z);
        }
    }

    __host__ __device__ inline bool operator==(const ATOMIC_Scene& Other) const
    {
        if (SceneSize != Other.SceneSize || ActorCount != Other.ActorCount)
        {
            return false;
        }
        for (int i = 0; i < ATOMIC_MAX_BLOCKS; ++i)
        {
            if (SceneBlock[i] != Other.SceneBlock[i])
            {
                return false;
            }
        }

        for (int i = 0; i < ATOMIC_MAX_ACTORS; ++i)
        {
            if (Actors[i] != Other.Actors[i])
            {
                return false;
            }
        }
        return true;
    }
    __host__ __device__ inline bool operator%=(const ATOMIC_Scene& Other) const
    {
        if (SceneSize != Other.SceneSize || ActorCount != Other.ActorCount)
        {
            return false;
        }
        for (int i = 0; i < ATOMIC_MAX_ACTORS; ++i)
        {
            if (Actors[i] != Other.Actors[i])
            {
                return false;
            }
        }
        return true;
    }
    __host__ __device__ bool operator<(const ATOMIC_Scene& Other) const 
    {
        if (SceneSize.x != Other.SceneSize.x)
        {
            return SceneSize.x < Other.SceneSize.x;
        }
        if (SceneSize.y != Other.SceneSize.y)
        {
            return SceneSize.y < Other.SceneSize.y;
        }
        if (SceneSize.z != Other.SceneSize.z)
        {
            return SceneSize.z < Other.SceneSize.z;
        }
        if (ActorCount != Other.ActorCount)
        {
            return ActorCount < Other.ActorCount;
        }
        for (int x = 0; x < SceneSize.x; x++)
        {
            for (int y = 0; y < SceneSize.y; y++)
            {
                for (int z = 0; z < SceneSize.z; z++)
                {
                    int Index = x + y * SceneSize.x + z * SceneSize.x * SceneSize.y;
                    if (SceneBlock[Index] != Other.SceneBlock[Index])
                    {
                        return SceneBlock[Index] < Other.SceneBlock[Index];
                    }
                }
            }
        }
        for (int j = 0; j < ActorCount; j++)
        {
            if (Actors[j] != Other.Actors[j])
            {
                return Actors[j] < Other.Actors[j];
            }
        }
        return false;
    }
};
//TODO:
//out ATOMIC_Scene[N], bool[N], in ATOMIC_Scene[N], int2 Moves[N]
//out ATOMIC_Scene[N], bool[N], in ATOMIC_Scene[M], int2 Moves[N], int MoveOnIndex[N] # ATOMIC_Scenes[MoveOnIndex[i]]

//BFS
/*
States[]
while States not empty do:
    NewState[] # len(NewState) = 4.0 * len(States)
    NewStateValid[] # len(NewStateValid) = 4.0 * len(States)
    do in every thread t:
        for i, m in AllMovement: # AllMovement: up down left right
            Valid, CandidateState = States[t].Move(m)
            NewState[i + 4*t] = CandidateState
            NewStateValid[i + 4*t] = Valid
    NewStateValid <- scan: NewStateValid[Valid]
    States <- remove_duplicate: NewStateValid
        
*/
