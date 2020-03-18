#include "DensifiedWtaHash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
#include <algorithm>
#include <map>
#include "Config.h"
using namespace std;


DensifiedWtaHash::DensifiedWtaHash(int numHashes, int noOfBitsToHash)
{

    _numhashes = numHashes;
    _rangePow = noOfBitsToHash;

    std::random_device rd;
    std::mt19937 gen(rd());

    _permute = ceil(_numhashes * binsize * 1.0 / noOfBitsToHash);

    int* n_array = new int[_rangePow];
    _indices = new int[_rangePow * _permute];
    _pos = new int[_rangePow * _permute];

    for (int i = 0; i < _rangePow; i++) {
        n_array[i] = i;
    }

    for (int p = 0; p < _permute ;p++) {
        std::shuffle(n_array, n_array + _rangePow, rd);
        for (int j = 0; j < _rangePow; j++) {
            _indices[p * _rangePow + n_array[j]] = (p * _rangePow + j) / binsize;
            _pos[p * _rangePow + n_array[j]] = (p * _rangePow + j)%binsize;
        }
    }
    delete [] n_array;

    _lognumhash = log2(numHashes);
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


int * DensifiedWtaHash::getHashEasy(float* data, int dataLen, int topk)
{
    // binsize is the number of times the range is larger than the total number of hashes we need.

    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    for (int p=0; p< _permute; p++) {
        int bin_index = p * _rangePow;
        for (int i = 0; i < dataLen; i++) {
            int inner_index = bin_index + i;
            int binid = _indices[inner_index];
            float loc_data = data[i];
            if(binid < _numhashes && values[binid] < loc_data) {
                values[binid] = loc_data;
                hashes[binid] = _pos[inner_index];
            }
        }
    }

    for (int i = 0; i < _numhashes; i++)
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
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
    return hashArray;
}

int* DensifiedWtaHash::getHash(int* indices, float* data, int dataLen)
{
    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    // init hashes and values to INT_MIN to start
    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }

    //
    for (int p = 0; p < _permute; p++) {
        for (int i = 0; i < dataLen; i++) {
            int binid = _indices[p * _rangePow + indices[i]];
            if(binid < _numhashes) {
                if (values[binid] < data[i]) {
                    values[binid] = data[i];
                    hashes[binid] = _pos[p * _rangePow + indices[i]];
                }
            }
        }
    }

    for (int i = 0; i < _numhashes; i++)
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
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }

    delete[] hashes;
    delete[] values;

    return hashArray;
}


int DensifiedWtaHash::getRandDoubleHash(int binid, int count) {
    unsigned int tohash = ((binid + 1) << 6) + count;
    return (_randHash[0] * tohash << 3) >> (32 - _lognumhash); // _lognumhash needs to be ceiled.
}


DensifiedWtaHash::~DensifiedWtaHash()
{
    delete[] _randHash;
    delete[] _indices;
}
