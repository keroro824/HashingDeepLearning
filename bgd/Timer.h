#include <chrono>

class Timer
{
private:
    int _ForwardMul;
    int _BackMul;
    int _Lsh;
    int _Wta;
    int _Sort;

public:
    Timer();
    std::chrono::high_resolution_clock::time_point start();
    void addForward(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending);
    void addBack(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending);
    void addSort(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending);
    void addLsh(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending);
    void addWta(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point ending);

    int getForwardMul();
    int getBackMul();
    int getSort();
    int getLsh();
    int getWta();

};