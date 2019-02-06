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
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash;
public:
    WtaHash(int numHashes, int noOfBitsToHash);
    int * getHash(int* indices, float* data, int* binids, int dataLen);
    int getRandDoubleHash(int binid, int count);
    int * getHashEasy(int* binids, float* data, int dataLen);
    void getMap(int n, int* binid);
    ~WtaHash();
};