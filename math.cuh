#pragma once
#include <cuda_runtime.h>
__host__ __device__ inline float step(float edge, float x)
{
    return x < edge ? 0.0f : 1.0f;
}
__host__ __device__ inline float3 step(float3 edge, float3 x)
{
    return make_float3(x.x < edge.x ? 0.0f : 1.0f, x.y < edge.y ? 0.0f : 1.0f, x.z < edge.z ? 0.0f : 1.0f);
}
__host__ __device__ inline float4 make_float4(float x, float3 yzw)
{
    return make_float4(x, yzw.x, yzw.y, yzw.z);
}
__host__ __device__ inline float4 make_float4(float3 xyz, float w)
{
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}
__host__ __device__ inline float sign(float x)
{
    return float(x > 0.0f) - float(x < 0.0f);
}
__host__ __device__ inline float3 abs(float3 x)
{
    return { abs(x.x), abs(x.y),abs(x.z) };
}
__host__ __device__ inline float3 sqrt(float3 x)
{
    return { sqrt(x.x), sqrt(x.y),sqrt(x.z) };
}
__host__ __device__ inline float length(float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__host__ __device__ inline float3 normalize(float3 v)
{
    float len = length(v);
    len = max(len, 1e-8f);
    return make_float3(v.x / len, v.y / len, v.z / len);
}
__host__ __device__ inline float3 cross(float3 a, float3 b)
{
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}
__host__ __device__ inline float3 operator+(float3 a, float3 b) {
    float3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

__host__ __device__ inline float3 operator-(float3 a, float3 b) 
{
    float3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}
__host__ __device__ inline float3 operator*(float scalar, float3 vec)
{
    float3 result;
    result.x = scalar * vec.x;
    result.y = scalar * vec.y;
    result.z = scalar * vec.z;
    return result;
}
__host__ __device__ inline float3 operator*(float3 vec1, float3 vec2)
{
    float3 result;
    result.x = vec1.x * vec2.x;
    result.y = vec1.y * vec2.y;
    result.z = vec1.z * vec2.z;
    return result;
}
__host__ __device__ inline bool operator==(float3 vec1, float3 vec2)
{
    return (vec1.x == vec2.x) && (vec1.y == vec2.y) && (vec1.z == vec2.z);
}
__host__ __device__ inline bool operator==(int3 vec1, int3 vec2)
{
    return (vec1.x == vec2.x) && (vec1.y == vec2.y) && (vec1.z == vec2.z);
}
__host__ __device__ inline bool operator!=(int3 vec1, int3 vec2)
{
    return !((vec1.x == vec2.x) && (vec1.y == vec2.y) && (vec1.z == vec2.z));
}
__host__ __device__ inline bool operator==(int2 vec1, int2 vec2)
{
    return (vec1.x == vec2.x) && (vec1.y == vec2.y);
}
__host__ __device__ inline bool operator!=(int2 vec1, int2 vec2)
{
    return !((vec1.x == vec2.x) && (vec1.y == vec2.y));
}
__host__ __device__ inline float3 operator/(float3 vec1, float3 vec2)
{
    float3 result;
    result.x = vec1.x / vec2.x;
    result.y = vec1.y / vec2.y;
    result.z = vec1.z / vec2.z;
    return result;
}
__host__ __device__ inline float dot(float3 vec1, float3 vec2)
{
    float3 result;
    result.x = vec1.x / vec2.x;
    result.y = vec1.y / vec2.y;
    result.z = vec1.z / vec2.z;
    return result.x + result.y + result.z;
}
__host__ __device__ inline float3 operator/(float scalar, float3 vec)
{
    float3 result;
    result.x = scalar / vec.x;
    result.y = scalar / vec.y;
    result.z = scalar / vec.z;
    return result;
}
__host__ __device__ inline float3 operator+(float scalar, float3 vec)
{
    float3 result;
    result.x = scalar + vec.x;
    result.y = scalar + vec.y;
    result.z = scalar + vec.z;
    return result;
}
__host__ __device__ inline float3 operator*(float3 vec, float scalar)
{
    return scalar * vec;
}
__host__ __device__ inline float3 operator/(float3 vec, float scalar)
{
    return scalar / vec;
}
__host__ __device__ inline float3 operator+(float3 vec, float scalar)
{
    return scalar + vec;
}
__host__ __device__ inline float max3(float3 x)
{
    return max(max(x.x, x.y), x.z);
}
__host__ __device__ inline float mul3(float3 x)
{
    return x.x * x.y * x.z;
}
__host__ __device__ inline float grid(float3 x)
{
    return max(max((x.x * x.y), (x.y * x.z)), (x.x * x.z));
}
__host__ __device__ inline float clamp(float x)
{
    return min(max(x, 0.0f), 1.0f);
}
__host__ __device__ inline float3 clamp(float3 x)
{
    return make_float3(clamp(x.x), clamp(x.y), clamp(x.z));
}