#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
#include <string.h>
#include "MurmurHash.h"
#include "Util.h"

/*
*  Algorithm from the paper The Power of Comparative Reasoning. Jay Yagnik, Dennis Strelow, David A. Ross, Ruei-sung Lin

*/
class WtaHash
{
private:
    int _numhashes, _rangePow;
    std::vector<int> _indices;
public:
    WtaHash(int numHashes, int noOfBitsToHash);
    std::vector<int> getHash(const std::vector<float> &data);
    std::vector<int> getHash(const SubVectorConst<float> &data);
    ~WtaHash();
};