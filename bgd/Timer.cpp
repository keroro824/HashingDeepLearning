#include "Timer.h"
#include <chrono>

using namespace std;

Timer::Timer()
{
    _ForwardMul = 0;
    _BackMul = 0;
    _Lsh = 0;
    _Wta = 0;
    _Sort = 0;
}


std::chrono::high_resolution_clock::time_point Timer::start(){
    return std::chrono::high_resolution_clock::now();
}

void Timer::addForward(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending)
{
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(ending - start).count();
    _ForwardMul += timeDiffInMiliseconds;

}

void Timer::addBack(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending)
{
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(ending - start).count();
    _BackMul += timeDiffInMiliseconds;
}

void Timer::addSort(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending)
{
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(ending - start).count();
    _Sort += timeDiffInMiliseconds;
}

void Timer::addLsh(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending)
{
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(ending - start).count();
    _Lsh += timeDiffInMiliseconds;
}


void Timer::addWta(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending)
{
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(ending - start).count();
    _Wta += timeDiffInMiliseconds;
}



int Timer::getForwardMul()
{
    return _ForwardMul;
}

int Timer::getBackMul()
{
    return _BackMul;
}

int Timer::getSort()
{
    return _Sort;
}

int Timer::getLsh()
{
    return _Lsh;
}

int Timer::getWta()
{
    return _Wta;
}