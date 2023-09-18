#pragma once
#include "scene.h"

#define ATOMIC_MAX_BLOCKS 16*16*16
#define ATOMIC_MAX_ACTORS 64
struct ATOMIC_Scene
{
public:
    uint8_t SceneBlock[ATOMIC_MAX_BLOCKS];
    Actor Actors[ATOMIC_MAX_ACTORS];
    int3 SceneSize;
    int ActorCount;

    bool InitialFromScene(const Scene& SourceScene)
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

    bool SetupScene(const vector<uint8_t>& SceneBlock_, const vector<Actor>& Actors_, int3 SceneSize_)
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
    void Reset()
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

    __host__ __device__ bool PositionOutOfBound(int3 Position) const
    {
        return Position.x < 0 || Position.x >= SceneSize.x || Position.y < 0 || Position.y >= SceneSize.y || Position.z < 0 || Position.z >= SceneSize.z;
    }
    __host__ __device__ bool MovePlayer(int2 Move)
    {
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
                if (PositionOutOfBound(NewPosition))
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
                            if (PositionOutOfBound(BoxNewPosition))
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
    vector<RenderData> GetRenderData()
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
};