#pragma once
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

#define SOKOBAN_WALL_COLOR {210.0f/255.0f, 180.0f/255.0f, 140.0f/255.0f}
#define SOKOBAN_PLAYER_START_COLOR {238.0f/255.0f, 130.0f/255.0f, 238.0f/255.0f}
#define SOKOBAN_PLAYER_COLOR {65.0f/255.0f, 105.0f/255.0f, 255.0f/255.0f}
#define SOKOBAN_BOX_COLOR {205.0f/255.0f, 133.0f/255.0f, 63.0f/255.0f}

struct Actor 
{
	int ActorType;
	int ActorState = 0;
	int3 Location;
	int3 Rotation;
	int Id = -1;
};

struct RenderData
{
	int3 Location;
	float3 TypeColor;
	float3 Scale;
	float3 Offset;
};

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
		SceneBlock = SceneBlock_;
		Actors = Actors_;
		SceneSize = SceneSize_;
	}
	void AddActor(Actor Actor)
	{
		if (Actor.ActorType == SOKOBAN_PLAYER)
		{
			for (auto& CurrentActor : Actors)
			{
				if (CurrentActor.ActorType == SOKOBAN_PLAYER_START && CurrentActor.ActorState == 0)
				{
					Actor.Location = CurrentActor.Location;
					CurrentActor.ActorState = 1;
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
							ActorW.Location = BoxNewPosition;
						}
					}
				}
				Actor.Location = NewPosition;
				return true;
			}
		}
		return false;
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
			uint8_t ActorsType = Actor.ActorType;
			int x = Actor.Location.x;
			int y = Actor.Location.y;
			int z = Actor.Location.z;
			if (ActorsType == SOKOBAN_PLAYER_START)
			{
				Result.push_back({ {x,y,z},SOKOBAN_PLAYER_START_COLOR , {1.0f,0.5f,1.0f}, {0.0f,0.5f,0.0f} });
			}
			else if (ActorsType == SOKOBAN_PLAYER)
			{
				Result.push_back({ {x,y,z},SOKOBAN_PLAYER_COLOR , {0.8f,0.8f,0.8f}, {0.0f,0.0f,0.0f} });
			}
			else if (ActorsType == SOKOBAN_BOX)
			{
				Result.push_back({ {x,y,z},SOKOBAN_BOX_COLOR , {0.95f,0.95f,0.95f}, {0.0f,0.025f,0.0f} });
			}
		}
		return Result;
	}
};