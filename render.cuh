#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "scene.h"
#include "vectorize_scene.cuh"
#include "math.cuh"

__global__ void RenderScene_UV(const RenderData* SceneData, int NumData, float3* Out_Buffer, int2 OutBufferSize, float Time);
__global__ void RenderScene(const RenderData* SceneData, int NumData, int3 SceneSize, float3* Out_Buffer, int2 OutBufferSize, float Time);

vector<float3> Launch_RenderScene(const vector<RenderData>& SceneData, int3 SceneSize, int2 BufferSize, float Time);