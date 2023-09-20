#pragma once
#include <iostream>
#include <chrono>
#include <string>

class Timer 
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> StartTime;

public:
    Timer() {
        Start();
    }

    void Start()
    {
        StartTime = std::chrono::high_resolution_clock::now();
    }

    double Reset(std::string AdditionalInfo = "", bool bPrintDuration = true)
    {
        auto EndTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> Duration = EndTime - StartTime;
        if (bPrintDuration)
        {
            std::cout << AdditionalInfo << " Duration: " << Duration.count() << " ms" << std::endl;
        }
        Start();
        return Duration.count();
    }
};