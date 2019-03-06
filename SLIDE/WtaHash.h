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
/*
*  Algorithm from the paper The Power of Comparative Reasoning. Jay Yagnik, Dennis Strelow, David A. Ross, Ruei-sung Lin

*/
using namespace std;
class WtaHash
{
private:
    int *_indices, _numhashes, _rangePow;
public:
    WtaHash(int numHashes, int noOfBitsToHash);
    int * getHash(float* data);
    ~WtaHash();
};