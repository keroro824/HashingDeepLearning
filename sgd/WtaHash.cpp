#include "WtaHash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
using namespace std;


WtaHash::WtaHash(int numHashes, int noOfBitsToHash)
{

    _numhashes = numHashes;
    _rangePow = noOfBitsToHash;
    _lognumhash = log2(numHashes);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, INT_MAX);

    _randa = dis(gen);
    if (_randa % 2 == 0)
        _randa++;
    _randHash = new int[2];
    _randHash[0] = dis(gen);
    if (_randHash[0] % 2 == 0)
        _randHash[0]++;
    _randHash[1] = dis(gen);
    if (_randHash[1] % 2 == 0)
        _randHash[1]++;

}


void WtaHash::getMap(int n, int* binids)
{
    int range = 1 << _rangePow;
    // binsize is the number of times the range is larger than the total number of hashes we need.
    int binsize = ceil(1.0*range / _numhashes);
    for (size_t i = 0; i < n; i++)
    {
        unsigned int h = i;
        h *= _randa;
        h ^= h >> 13;
        h *= 0x85ebca6b;
        unsigned int curhash = (unsigned int)(((unsigned int)h*i) << 5);
//        uint32_t curhash = MurmurHash ((char *)indices, (uint32_t) sizeof(indices[i]), (uint32_t)_randa);
        curhash = curhash & ((1<<_rangePow)-1);
        binids[i] = (int)floor(curhash / binsize);;
    }

}


int * WtaHash::getHashEasy(int* binids, float* data, int dataLen)
{

    // binsize is the number of times the range is larger than the total number of hashes we need.

    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (size_t i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }


    for (size_t i = 0; i < dataLen; i++)
    {
        if (values[binids[i]] < data[i]) {
            values[binids[i]] = data[i];
            hashes[binids[i]] = i;
        }
    }

    for (size_t i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 10) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
    return hashArray;
}


int * WtaHash::getHash(int* indices, float* data, int* binids, int dataLen)
{
    // int dataLen = data.size();

//    int range = 1 << _rangePow;
    // binsize is the number of times the range is larger than the total number of hashes we need.
//    int binsize = ceil(1.0*range / _numhashes);

    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (size_t i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    if (dataLen<0){

    }

    for (size_t i = 0; i < dataLen; i++)
    {
        int binid = binids[indices[i]];

        if (values[binid] < data[i]){
            values[binid] = data[i];
            hashes[binid] = indices[i];
        }
    }

    for (size_t i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 10) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
    return hashArray;
}



int WtaHash::getRandDoubleHash(int binid, int count) {
    unsigned int tohash = ((binid + 1) << 6) + count;
    return (_randHash[0] * tohash << 3) >> (32 - _lognumhash); // _lognumhash needs to be ceiled.
}



WtaHash::~WtaHash()
{
    delete[] _randHash;
}