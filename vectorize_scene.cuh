#pragma once
#include "scene.h"
#include "math.cuh"
#include <string>

#define ATOMIC_MAX_BLOCKS 16*16*16
#define ATOMIC_MAX_ACTORS 16

//4 is enough for wsad and zero move
#define MOVEMENT_BIT 4
#define ATOMIC_MAX_STEP 255


struct ATOMIC_Steps
{
public:
    unsigned char Step[ATOMIC_MAX_STEP / (8 / MOVEMENT_BIT) + 1];
    unsigned char StepCount = 0;
    __host__ __device__ bool AddStep(int2 MoveDirection)
    {
        unsigned char NewIndex = GetIndexByStep(MoveDirection);
        if (StepCount < ATOMIC_MAX_STEP)
        {
            unsigned char& TargetChar = Step[StepCount / (8 / MOVEMENT_BIT)];

            if (StepCount % 2 == 0)
            {
                TargetChar &= 0x0F;
                TargetChar |= (NewIndex << 4);
            }
            else
            {
                TargetChar &= 0xF0;
                TargetChar |= NewIndex;
            }
            StepCount++;
            return true;
        }
        return false;
    }
    __host__ __device__ static ATOMIC_Steps GetEmptyStep()
    {
        ATOMIC_Steps Result;
        Result.Step[0] = 0;
        return Result;
    }
    __host__ __device__ int2 GetStep(unsigned int Index)
    {
        if (Index >= StepCount || Index >= ATOMIC_MAX_STEP)
        {
            return { 0, 0 };
        }
        unsigned char TargetChar = Step[Index / 2];
        unsigned char ExtractedIndex;

        if (Index % 2 == 0)
        {
            ExtractedIndex = (TargetChar & 0xF0) >> 4;
        }
        else
        {
            ExtractedIndex = TargetChar & 0x0F;
        }

        return GetStepByIndex(ExtractedIndex);
    }
    __host__ __device__ static int2 GetStepByIndex(unsigned char i)
    {
        int2 Step = { 0,0 };
        if (i == 1)
        {
            Step = { 1,0 };
        }
        else if (i == 2)
        {
            Step = { -1,0 };
        }
        else if (i == 3)
        {
            Step = { 0,1 };
        }
        else if (i == 4)
        {
            Step = { 0,-1 };
        }
        return Step;
    }
    __host__ __device__ static unsigned char GetIndexByStep(int2 MoveDirection)
    {
        if (MoveDirection.x > 0)
        {
            return 1u;
        }
        else if (MoveDirection.x < 0)
        {
            return 2u;
        }
        else if (MoveDirection.y > 0)
        {
            return 3u;
        }
        else if (MoveDirection.y < 0)
        {
            return 4u;
        }
        return 0u;
    }
};

struct STATIC_SceneBlock
{
public:
    uint8_t SceneBlock[ATOMIC_MAX_BLOCKS];
    int3 SceneSize = { 0,0,0 };
    __host__ bool InitialFromScene(const Scene& SourceScene)
    {
        for (size_t i = 0; i < SourceScene.SceneBlock.size(); ++i)
        {
            SceneBlock[i] = SourceScene.SceneBlock[i];
        }
        SceneSize = SourceScene.SceneSize;
        return true;
    }
    __host__ bool SetupScene(const vector<uint8_t>& SceneBlock_, int3 SceneSize_)
    {
        assert(SceneBlock_.size() == SceneSize_.x * SceneSize_.y * SceneSize_.z);
        if (SceneBlock_.size() > ATOMIC_MAX_BLOCKS)
        {
            return false;
        }
        SceneSize = SceneSize_;
        for (size_t i = 0; i < SceneBlock_.size(); ++i)
        {
            SceneBlock[i] = SceneBlock_[i];
        }
    }
    __host__ void Reset()
    {
        for (int i = 0; i < ATOMIC_MAX_BLOCKS; i++)
        {
            SceneBlock[i] = 0;
        }
        SceneSize = int3();
    }
    __host__ __device__ inline bool bIsPositionOutOfBound(int3 Position) const
    {
        return Position.x < 0 || Position.x >= SceneSize.x || Position.y < 0 || Position.y >= SceneSize.y || Position.z < 0 || Position.z >= SceneSize.z;
    }

    __host__ __device__ inline bool operator==(const STATIC_SceneBlock& Other) const
    {
        if (SceneSize != Other.SceneSize)
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
        return true;
    }
};

struct ATOMIC_Scene
{
public:
    Actor Actors[ATOMIC_MAX_ACTORS];
    unsigned char ActorCount = 0;
    unsigned char SceneIndex = 0;

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
        for (size_t i = 0; i < SourceScene.Actors.size(); ++i)
        {
            Actors[i] = SourceScene.Actors[i];
        }
        ActorCount = static_cast<unsigned char>(SourceScene.Actors.size());
        return true;
    }

    __host__ bool SetupScene(const vector<Actor>& Actors_)
    {
        if (Actors_.size() > ATOMIC_MAX_ACTORS)
        {
            return false;
        }
        ActorCount = static_cast<unsigned char>(Actors_.size());
        for (size_t i = 0; i < Actors_.size(); ++i)
        {
            Actors[i] = Actors_[i];
        }
        return true;
    }
    __host__ void Reset()
    {
        Actor defaultActor;
        for (int i = 0; i < ATOMIC_MAX_ACTORS; i++)
        {
            Actors[i] = defaultActor;
        }
        ActorCount = 0;
    }
    //TODO: Finalize scene, store the key value as static(after finalize)
    __host__ __device__ bool MovePlayer(int2 Move, const STATIC_SceneBlock& StaticSceneBlock)
    {
        if (Move.x == 0 && Move.y == 0)
        {
            return false;
        }
        int AbsMoveX = abs(Move.x);
        if (AbsMoveX == 1 && AbsMoveX == abs(Move.y)) 
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
                if (StaticSceneBlock.bIsPositionOutOfBound(NewPosition))
                {
                    return false;
                }
                int NewIndex = NewPosition.x + NewPosition.y * StaticSceneBlock.SceneSize.x + NewPosition.z * StaticSceneBlock.SceneSize.x * StaticSceneBlock.SceneSize.y;
                if (StaticSceneBlock.SceneBlock[NewIndex] == SOKOBAN_WALL)
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
                            if (StaticSceneBlock.bIsPositionOutOfBound(BoxNewPosition))
                            {
                                return false;
                            }
                            int BoxNewIndex = BoxNewPosition.x + BoxNewPosition.y * StaticSceneBlock.SceneSize.x + BoxNewPosition.z * StaticSceneBlock.SceneSize.x * StaticSceneBlock.SceneSize.y;
                            if (StaticSceneBlock.SceneBlock[BoxNewIndex] == SOKOBAN_WALL) // If box forward is wall
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
                            ActorW.Location = int3_2_uchar3(BoxNewPosition);
                        }
                    }
                }
                CurrentActor.Location = int3_2_uchar3(NewPosition);
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
                int3 BoxPosition = uchar3_2_int3(ActorW.Location);

                bool Continue = false;
                for (int j = 0; j < ActorCount; j++)
                {
                    Actor& ActorW2 = Actors[j];
                    if (ActorW2.ActorType == SOKOBAN_BOX_TARGET)
                    {
                        int3 TargetPosition = uchar3_2_int3(ActorW2.Location);
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
    vector<RenderData> GetRenderData(const STATIC_SceneBlock& StaticSceneBlock) const
    {
        vector<RenderData> Result;
        for (int x = 0; x < StaticSceneBlock.SceneSize.x; x++)
        {
            for (int y = 0; y < StaticSceneBlock.SceneSize.y; y++)
            {
                for (int z = 0; z < StaticSceneBlock.SceneSize.z; z++)
                {
                    int Index = x + y * StaticSceneBlock.SceneSize.x + z * StaticSceneBlock.SceneSize.x * StaticSceneBlock.SceneSize.y;
                    uint8_t BlockType = StaticSceneBlock.SceneBlock[Index];
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
    void Debug(const STATIC_SceneBlock& StaticSceneBlock)
    {
        for (int z = 0; z < StaticSceneBlock.SceneSize.z; z++)
        {
            for (int y = 0; y < StaticSceneBlock.SceneSize.y; y++)
            {
                for (int x = 0; x < StaticSceneBlock.SceneSize.x; x++)
                {
                    int Index = x + y * StaticSceneBlock.SceneSize.x + z * StaticSceneBlock.SceneSize.x * StaticSceneBlock.SceneSize.y;
                    printf("%d ", StaticSceneBlock.SceneBlock[Index]);
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
        if (ActorCount != Other.ActorCount)
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
};


struct ATOMIC_SolverState
{
    ATOMIC_Scene SceneState;
    ATOMIC_Steps StepState;
    bool WinState = false;
    bool ValidState = false;
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
