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


using namespace std;
class DensifiedMinhash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash;
public:
    DensifiedMinhash(int numHashes, int noOfBitsToHash);
    int * getHash(int* indices, float* data, int* binids, int dataLen);
    int getRandDoubleHash(int binid, int count);
    int * getHashEasy(int* binids, float* data, int dataLen, int topK);
    void getMap(int n, int* binid);
    ~DensifiedMinhash();
};