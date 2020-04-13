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
    int * getHash(const int* indices, const float* data, const std::vector<int> &binids, int dataLen);
    int getRandDoubleHash(int binid, int count);
    int * getHashEasy(const std::vector<int> &binids, const float* data, int dataLen, int topK);
    void getMap(int n, std::vector<int> &binid);
    ~DensifiedMinhash();
};