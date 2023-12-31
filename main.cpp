#include <iostream>
#include <vector>
#include <windows.h>
#include "render.cuh"
#include "solver.cuh"
#include "solver_gpu.cuh"
#include "scene.h"

#include <windows.h>

#include <thread>
#include <mutex>
#include "math.cuh"
#include "timer.h"
#include "vectorize_scene.cuh"

#include "test_scene.hpp"
using namespace std;

std::mutex SceneMutex;
std::mutex InputMutex;
int2 PlayerInput = {0,0};

vector<ATOMIC_Steps> AutoMoves;
Timer SolverTimer;
unsigned int CurrentMove = 0;
void CopyBufferToHDC(HDC& TargetHDC, const vector<float3>& ImageBuffer, int ImageSizeW, int ImageSizeH)
{
    for (int x = 0; x < ImageSizeW; ++x)
    {
        for (int y = 0; y < ImageSizeH; ++y)
        {
            float3 PRGB = ImageBuffer[x + y * ImageSizeW];
            uint8_t R = static_cast<uint8_t>(PRGB.x * 255);
            uint8_t G = static_cast<uint8_t>(PRGB.y * 255);
            uint8_t B = static_cast<uint8_t>(PRGB.z * 255);
            SetPixel(TargetHDC, x, y, RGB(R, G, B));
        }
    }
}
void RenderThreadFunction(ATOMIC_Scene& CurrentScene, const STATIC_SceneBlock& SceneBlock, int ImageSizeW, int ImageSizeH, HDC& TempDC, HDC& TargetHDC)
{
    float Time = 0.0f;
    while (true) 
    {
        SceneMutex.lock();
        /*
        bool Changed = CurrentScene.MovePlayer(PlayerInput);
        if (Changed)
        {
            printf("Moved:%d,%d\n", PlayerInput.x, PlayerInput.y);
            CurrentScene.UpdatePhysics();
            if (CurrentScene.bIsWin())
            {
                printf("Win\n");
            }
        }
        */
        if (AutoMoves.size() > 0)
        {
            if (CurrentMove < AutoMoves[0].StepCount)
            {
                PlayerInput = AutoMoves[0].GetStep(CurrentMove);
                CurrentMove++;
                bool Changed = CurrentScene.MovePlayer(PlayerInput, SceneBlock);
                if (Changed)
                {
                    printf("Moved:%d,%d\n", PlayerInput.x, PlayerInput.y);
                    CurrentScene.UpdatePhysics();
                    if (CurrentScene.bIsWin())
                    {
                        printf("Win\n");
                    }
                }
            }
        }
        SceneMutex.unlock();

        SceneMutex.lock();
        vector<float3> ImageBuffer = Launch_RenderScene(CurrentScene.GetRenderData(SceneBlock), SceneBlock.SceneSize, { ImageSizeW , ImageSizeH }, Time);
        CopyBufferToHDC(TempDC, ImageBuffer, ImageSizeW, ImageSizeH);
        BitBlt(TargetHDC, 512, 64, ImageSizeW, ImageSizeH, TempDC, 0, 0, SRCCOPY);
        SceneMutex.unlock();

        InputMutex.lock();
        PlayerInput = { 0, 0 };
        InputMutex.unlock();
    }
}

bool IsKeyPressed(int vKey) 
{
    return GetAsyncKeyState(vKey) & 0x8000;
}
int2 GetCurrentInputAxis()
{
    int IntX = IsKeyPressed(VK_RIGHT) ? 1 : (IsKeyPressed(VK_LEFT) ? -1 : 0);
    int IntY = IsKeyPressed(VK_UP) ? 1 : (IsKeyPressed(VK_DOWN) ? -1 : 0);
    return { IntX, IntY };
}
int main()
{
    HDC TargetHDC = GetDC(GetConsoleWindow());
    HDC TempDC = CreateCompatibleDC(TargetHDC);

    int ImageSizeW = 384;
    int ImageSizeH = 384;

    HBITMAP HBitmap = CreateCompatibleBitmap(TargetHDC, ImageSizeW, ImageSizeH);
    HBITMAP HOldBitmap = (HBITMAP)SelectObject(TempDC, HBitmap);

    Scene TestScene = GetHardScene(); // GetSimpleScene();
    ATOMIC_Scene AtomicScene;
    AtomicScene.InitialFromScene(TestScene);
    STATIC_SceneBlock StaticSceneBlock;
    StaticSceneBlock.InitialFromScene(TestScene);
    //AtomicScene.Debug();
    int N_SolveSceneNum = 1;
    SolverTimer.Start();
    for (int i = 0; i < N_SolveSceneNum; i++)
    {
        AutoMoves = CPU_Solver::Solve(AtomicScene, StaticSceneBlock, true);
    }
    SolverTimer.Reset(string("CPU Solver"), true);
    printf("All Possible: %zd\n", AutoMoves.size());
    
    vector<ATOMIC_Scene> InitialScene;
    vector<STATIC_SceneBlock> SceneBlocks;
    for (int i = 0; i < N_SolveSceneNum; i++)
    {
        InitialScene.push_back(AtomicScene);
        SceneBlocks.push_back(StaticSceneBlock);
    }
    SolverTimer.Start();
    AutoMoves = GPU_Solver::Solve(InitialScene, SceneBlocks, true);
    SolverTimer.Reset(string("GPU Solver"), true);
    printf("All Possible: %zd\n", AutoMoves.size());
    /*
    while (true)
    {
        //Input
        int IntX = IsKeyPressed(VK_LEFT) ? -1 : 0;
        IntX += IsKeyPressed(VK_RIGHT) ? 1 : 0;
        int IntY = IsKeyPressed(VK_DOWN) ? -1 : 0;
        IntY += IsKeyPressed(VK_UP) ? 1 : 0;
        //printf("%d, %d\n", IntX, IntY);
        TestScene.MovePlayer({IntX, IntY});
        //
        vector<float3> ImageBuffer = Launch_RenderScene(TestScene.GetRenderData(), SceneSize, { ImageSizeW , ImageSizeH }, Time);
        CopyBufferToHDC(TempDC, ImageBuffer, ImageSizeW, ImageSizeH);
        BitBlt(TargetHDC, 0, 64, ImageSizeW, ImageSizeH, TempDC, 0, 0, SRCCOPY);
        //Time += 0.125f;
    }
    */
    std::thread RenderThread(RenderThreadFunction, std::ref(AtomicScene), std::ref(StaticSceneBlock), ImageSizeW, ImageSizeH, std::ref(TempDC), std::ref(TargetHDC));
    while (true) 
    {
        if (AutoMoves.size() > 0)
        {
            InputMutex.lock();
            PlayerInput = (PlayerInput == int2{ 0,0 }) ? GetCurrentInputAxis() : PlayerInput;
            InputMutex.unlock();
        }
        //cout << PlayerInput.x << ", " << PlayerInput.y << "\n";
    }
    RenderThread.join();

    SelectObject(TempDC, HOldBitmap);
    DeleteObject(HBitmap);
    DeleteDC(TempDC);
    ReleaseDC(GetConsoleWindow(), TargetHDC);

    getchar();
    return 0;
}