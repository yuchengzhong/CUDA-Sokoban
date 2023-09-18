#include "solver.cuh"

__device__ float4 iBox(float3 ro, float3 rd, float3 rad, float3 bo, float3& LocalPosition) //return nor, tn
{
    //vec3 rdd = (txx * vec4(rd, 0.0)).xyz;  // txx the W2O
    //vec3 roo = (txx * vec4(ro, 1.0)).xyz;
    float3 roo = ro - bo;
    float3 rdd = rd;  // txx the W2O

    // ray-box intersection in box space
    float3 m = { 1.0f / rdd.x,  1.0f / rdd.y, 1.0f / rdd.z };

    // more robust
    float3 k = make_float3(rdd.x >= 0.0 ? rad.x : -rad.x, rdd.y >= 0.0 ? rad.y : -rad.y, rdd.z >= 0.0 ? rad.z : -rad.z);
    //float3 t1 = (-roo - k) * m;
    //float3 t2 = (-roo + k) * m;
    float3 t1 = { (-roo.x - k.x) * m.x, (-roo.y - k.y) * m.y, (-roo.z - k.z) * m.z };
    float3 t2 = { (-roo.x + k.x) * m.x, (-roo.y + k.y) * m.y, (-roo.z + k.z) * m.z };

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);

    // no intersection
    if (tN > tF || tF < 0.0)
    {
        return make_float4(-1.0, -1.0, -1.0, -1.0);
    }
    float4 res = (tN > 0.0) ? make_float4(step(make_float3(tN, tN, tN), t1), tN) : make_float4(step(t2, make_float3(tF, tF, tF)), tF);
    //res.yzw = (txi * vec4(-sign(rdd) * res.yzw, 0.0)).xyz; // txi the O2W
    //float3 nor = -sign(rdd) * res.xyz;
    res.x = -sign(rdd.x) * res.x;
    res.y = -sign(rdd.y) * res.y;
    res.z = -sign(rdd.z) * res.z;

    LocalPosition = (res.w * rdd + roo) / rad;
    return res;
}


__global__ void RenderScene_UV(const RenderData* SceneData, int NumData, float3* Out_Buffer, int2 OutBufferSize, float Time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= OutBufferSize.x || y >= OutBufferSize.y)
    {
        return;
    }
    int Index = y * OutBufferSize.x + x;

    float3 PixelColor = { float(x) / OutBufferSize.x, float(y) / OutBufferSize.y, sin(Time) * 0.5f + 0.5f};

    Out_Buffer[Index] = PixelColor;
}

__global__ void RenderScene(const RenderData* SceneData, int NumData, int3 SceneSize, float3* Out_Buffer, int2 OutBufferSize, float Time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= OutBufferSize.x || y >= OutBufferSize.y)
    {
        return;
    }
    int Index = y * OutBufferSize.x + x;

    float2 p = make_float2((float(x) + 0.5f) / OutBufferSize.x - 0.5f, (float(y) + 0.5f) / OutBufferSize.y - 0.5f);

    // camera movement	
    float an = 0.4f * Time + 3.14f + 3.14f / 2.0f;
    float length = 3.0f;
    float3 ro = make_float3(length * cos(an), -1.75*length, length * sin(an));
    float3 ta = make_float3(0.0f, 0.25f, 0.0f);
    // camera matrix
    float3 ww = normalize(ta - ro);
    float3 uu = normalize(cross(ww, make_float3(0.0, 1.0, 0.0)));
    float3 vv = normalize(cross(uu, ww));
    // create view ray
    float3 rd = normalize(p.x * uu + p.y * vv + 2.0 * ww);
    // GBuffer
    float3 PixelGBufferAlbedo = make_float3(0.0f, 0.0f, 0.0f);
    float3 PixelGBufferNormal = make_float3(0.0f, 0.0f, 0.0f);
    float3 PixelGBufferPosition = make_float3(0.0f, 0.0f, 0.0f);
    //
    float MinDepth = 1e8f;
    float MaxSceneSize = max(max(SceneSize.x, SceneSize.y), SceneSize.z);
    float3 HalfBoxSize = 1.0f / make_float3(MaxSceneSize, MaxSceneSize, MaxSceneSize);
    float3 BoxSize = HalfBoxSize * 2.0f;
    for (int i = 0; i < NumData; i++)
    {
        //
        RenderData CurrentBox = SceneData[i];
        float3 IntPosition = make_float3(SceneSize.x - CurrentBox.Location.x - 1, -CurrentBox.Location.z + (MaxSceneSize - SceneSize.z), CurrentBox.Location.y); //[0, N)
        float3 CurrentBoxPosition = (IntPosition + 0.5f + CurrentBox.Offset) * BoxSize - make_float3(1.0f, 1.0f, 1.0f);//[0.5, N + 0.5), [0.5/N, 1.0 + 0.5/N), [0.5/N, 1.0 + 0.5/N)
        //
        float3 LocalPosition;
        float4 BoxResult = iBox(ro, rd, HalfBoxSize * CurrentBox.Scale, CurrentBoxPosition, LocalPosition);
        if (BoxResult.w > 0.0f && BoxResult.w < MinDepth)
        {
            float3 Normal = make_float3(BoxResult.x, BoxResult.y, BoxResult.z);
            MinDepth = BoxResult.w;
            PixelGBufferNormal = Normal;

            float GridAlpha = 1.0f - step(0.9f, grid(abs(LocalPosition)));
            PixelGBufferAlbedo = CurrentBox.TypeColor* GridAlpha;

            PixelGBufferPosition = BoxResult.w * rd + ro;
        }
    }
    // Shadow ray
    float3 LightDirection = normalize({ 0.85f, -1.0f, 0.25f });
    float MinHitDistance = 1e8f;
    for (int i = 0; i < NumData; i++)
    {
        //
        RenderData CurrentBox = SceneData[i];
        float3 IntPosition = make_float3(SceneSize.x - CurrentBox.Location.x - 1, -CurrentBox.Location.z + (MaxSceneSize - SceneSize.z), CurrentBox.Location.y); //[0, N)
        float3 CurrentBoxPosition = (IntPosition + 0.5f + CurrentBox.Offset) * BoxSize - make_float3(1.0f, 1.0f, 1.0f);//[0.5, N + 0.5), [0.5/N, 1.0 + 0.5/N), [0.5/N, 1.0 + 0.5/N)
        //
        float3 LocalPosition;
        float4 BoxResult = iBox(PixelGBufferPosition + PixelGBufferNormal * 1e-6f, LightDirection, HalfBoxSize * CurrentBox.Scale, CurrentBoxPosition, LocalPosition);
        if (BoxResult.w > 0.0f && BoxResult.w < MinHitDistance)
        {
            MinHitDistance = BoxResult.w;
        }
    }
    float NdL = max(0.0f, dot(make_float3(PixelGBufferNormal.x, -PixelGBufferNormal.z, -PixelGBufferNormal.y), LightDirection));
    float Shadow = 1.0f - exp(-0.1f * MinHitDistance);
    float SkyLight = 0.125f;
    float3 PixelColor = PixelGBufferAlbedo *(NdL * Shadow * 0.5f + SkyLight);
    Out_Buffer[Index] = clamp(sqrt(PixelColor * 0.5f));
}

vector<float3> Launch_RenderScene(const vector<RenderData>& SceneData, int3 SceneSize, int2 BufferSize, float Time)
{
    RenderData* Device_SceneData;
    size_t SceneDataNumBytes = SceneData.size() * sizeof(RenderData);

    cudaMalloc((void**)&Device_SceneData, SceneDataNumBytes);
    cudaMemcpy(Device_SceneData, SceneData.data(), SceneDataNumBytes, cudaMemcpyHostToDevice);

    float3* Device_OutBuffer;
    cudaMalloc((void**)&Device_OutBuffer, BufferSize.x * BufferSize.y * sizeof(float3));

    // Launch kernel
    dim3 BlockSize(8, 8);
    dim3 GridSize((BufferSize.x + BlockSize.x - 1) / BlockSize.x, (BufferSize.y + BlockSize.y - 1) / BlockSize.y);

    RenderScene << <GridSize, BlockSize >> > (Device_SceneData, int(SceneData.size()), SceneSize, Device_OutBuffer, BufferSize, Time);
    cudaDeviceSynchronize();

    // Copy results back
    vector<float3> Result(BufferSize.x * BufferSize.y);
    cudaMemcpy(Result.data(), Device_OutBuffer, BufferSize.x * BufferSize.y * sizeof(float3), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(Device_SceneData);
    cudaFree(Device_OutBuffer);

    return Result;
}