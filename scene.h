#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include "math.cuh"
using namespace std;

#define SOKOBAN_EMPTY 0
#define SOKOBAN_WALL 1
#define SOKOBAN_PLAYER_START 2
#define SOKOBAN_PLAYER 3
#define SOKOBAN_BOX 4
#define SOKOBAN_BOX_TARGET 5

#define SOKOBAN_WALL_COLOR float3{210.0f/255.0f, 180.0f/255.0f, 140.0f/255.0f}
#define SOKOBAN_PLAYER_START_COLOR float3{238.0f/255.0f, 130.0f/255.0f, 238.0f/255.0f}
#define SOKOBAN_PLAYER_COLOR float3{65.0f/255.0f, 105.0f/255.0f, 255.0f/255.0f}

#define SOKOBAN_BOX_COLOR float3{205.0f/255.0f, 133.0f/255.0f, 63.0f/255.0f}
#define SOKOBAN_BOX_COLOR_ACTIVE float3{124.0f/255.0f, 252.0f/255.0f, 0.0f/255.0f}

#define SOKOBAN_BOX_TARGET_COLOR float3{173.0f/255.0f, 255.0f/255.0f, 47.0f/255.0f}

#define SOKOBAN_ERROR_COLOR float3{255.0f/255.0f, 255.0f/255.0f, 0.0f/255.0f}

#define SOKOBAN_DEFAULT_INT3 int3{ 0, 0, 0 }
#define SOKOBAN_DEFAULT_UCHAR3 uchar3{ 0u, 0u, 0u }
#define SOKOBAN_DEFAULT_FLOAT3 float3{ -999.0f,-999.0f,-999.0f }

#define SOKOBAN_INACTIVE 0
#define SOKOBAN_ACTIVE 1

#define SOKOBAN_ACTOR_DEFAULT_ROTATION 0
#define SOKOBAN_ACTOR_DEFAULT_ID 255
struct Actor
{
	unsigned char ActorType = 255;
	unsigned char ActorState = 0;
	uchar3 Location = SOKOBAN_DEFAULT_UCHAR3;
	//unsigned char Rotation = SOKOBAN_ACTOR_DEFAULT_ROTATION; //Move this to ActorState
	unsigned char Id = SOKOBAN_ACTOR_DEFAULT_ID;

	__host__ __device__ inline bool operator!=(const Actor& ActorOther) const
	{
		return (ActorOther.ActorType != ActorType) ||
			(ActorOther.ActorState != ActorState) ||
			(ActorOther.Location != Location) ||
			//(ActorOther.Rotation != Rotation) ||
			(ActorOther.Id != Id);
	}
	__host__ __device__ inline bool operator==(const Actor& ActorOther) const
	{
		return (ActorOther.ActorType == ActorType) &&
			(ActorOther.ActorState == ActorState) &&
			(ActorOther.Location == Location) &&
			//(ActorOther.Rotation == Rotation) &&
			(ActorOther.Id == Id);
	}
};

struct RenderData
{
	int3 Location = SOKOBAN_DEFAULT_INT3;
	float3 TypeColor = SOKOBAN_ERROR_COLOR;
	float3 Scale = SOKOBAN_DEFAULT_FLOAT3;
	float3 Offset = SOKOBAN_DEFAULT_FLOAT3;
};




// Deprecated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Deprecated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Deprecated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Deprecated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class Scene
{
public:
	//0 represent empty, 1 represent wall 
	vector<uint8_t> SceneBlock;
	vector<Actor> Actors;
	int3 SceneSize;
	int ActorId = 0;

	void SetupScene(const vector<uint8_t>& SceneBlock_, const vector<Actor>& Actors_, int3 SceneSize_)
	{
		assert(SceneBlock_.size() == SceneSize_.x * SceneSize_.y * SceneSize_.z);
		Actors = Actors_;
		SceneBlock.resize(SceneBlock_.size());
		SceneSize = SceneSize_;
		for (uint8_t x = 0; x < SceneSize.x; x++)
		{
			for (uint8_t y = 0; y < SceneSize.y; y++)
			{
				for (uint8_t z = 0; z < SceneSize.z; z++)
				{
					int Index = x + y * SceneSize.x + z * SceneSize.x * SceneSize.y;
					uint8_t BlockType = SceneBlock_[Index];
					if (BlockType == SOKOBAN_WALL)
					{
						SceneBlock[Index] = BlockType;
					}
					else if (BlockType == SOKOBAN_EMPTY)
					{
						SceneBlock[Index] = BlockType;
					}
					else if (BlockType == SOKOBAN_PLAYER)
					{
						AddActor({ SOKOBAN_PLAYER_START, 0, {x, y, z}, SOKOBAN_ACTOR_DEFAULT_ID });
						AddActor({ SOKOBAN_PLAYER, 0, {x, y, z}, SOKOBAN_ACTOR_DEFAULT_ID });
						std::cout << Actors.size() << "\n";
					}
					else if (BlockType == SOKOBAN_BOX)
					{
						AddActor({ SOKOBAN_BOX, 0, {x, y, z}, SOKOBAN_ACTOR_DEFAULT_ID });
						std::cout << Actors.size() << "\n";
					}
					else if (BlockType == SOKOBAN_BOX_TARGET)
					{
						AddActor({ SOKOBAN_BOX_TARGET, 0, {x, y, z}, SOKOBAN_ACTOR_DEFAULT_ID });
						std::cout << Actors.size() << "\n";
					}

				}
			}
		}
	}
	void AddActor(Actor Actor)
	{
		if (Actor.ActorType == SOKOBAN_PLAYER)
		{
			for (auto& CurrentActor : Actors)
			{
				if (CurrentActor.ActorType == SOKOBAN_PLAYER_START && CurrentActor.ActorState == Actor.ActorState)
				{
					Actor.Location = CurrentActor.Location;
				}
			}
		}
		Actor.Id = ActorId;
		ActorId++;
		Actors.push_back(Actor);
	}
	bool PositionOutOfBound(int3 Position) const
	{
		return Position.x < 0 || Position.x >= SceneSize.x || Position.y < 0 || Position.y >= SceneSize.y || Position.z < 0 || Position.z >= SceneSize.z;
	}
	bool MovePlayer(int2 Move)
	{
		if (abs(Move.x) == abs(Move.y) && abs(Move.x) == 1)
		{
			Move.y = 0;
		}
		for (auto& Actor : Actors)
		{
			uint8_t ActorsType = Actor.ActorType;
			if (ActorsType == SOKOBAN_PLAYER)
			{
				int3 NewPosition = { Actor.Location.x + Move.x, Actor.Location.y + Move.y, Actor.Location.z };
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
				for (auto& ActorW : Actors)
				{
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
							for (auto& ActorW2 : Actors) // If box forward is one more box
							{
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
				Actor.Location = int3_2_uchar3(NewPosition);
				return true;
			}
		}
		return false;
	}
	void UpdatePhysics()
	{
		//Do physics

		//Do after physics

		//Find box
		for (auto& ActorW : Actors)
		{
			if (ActorW.ActorType == SOKOBAN_BOX)
			{
				int3 BoxPosition = uchar3_2_int3(ActorW.Location);

				for (auto& ActorW2 : Actors)
				{
					if (ActorW2.ActorType == SOKOBAN_BOX_TARGET)
					{
						int3 TargetPosition = uchar3_2_int3(ActorW2.Location);
						if (TargetPosition == BoxPosition)
						{
							ActorW.ActorState = SOKOBAN_ACTIVE;
							goto FOR_ELSE;
						}
					}
				}
				ActorW.ActorState = SOKOBAN_INACTIVE;
				FOR_ELSE:
				{
					continue;
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
		for (auto& Actor : Actors)
		{
			int ActorsType = Actor.ActorType;
			int ActorsState = Actor.ActorState;
			int x = Actor.Location.x;
			int y = Actor.Location.y;
			int z = Actor.Location.z;
			if (ActorsType == SOKOBAN_PLAYER_START)
			{
				Result.push_back({ {x,y,z},SOKOBAN_PLAYER_START_COLOR , {1.0f,0.125f,1.0f}, {0.0f,0.5f-0.0625f,0.0f} });
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
		return Result;
	}
};