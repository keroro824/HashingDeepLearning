#include "WtaHash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
#include <algorithm>
#include <map>
#include "Config.h"
using namespace std;


WtaHash::WtaHash(int numHashes, int noOfBitsToHash)
{

    _numhashes = numHashes;
    _rangePow = noOfBitsToHash;

    std::random_device rd;
    std::mt19937 gen(rd());

    int permute = ceil(_numhashes*binsize*1.0/noOfBitsToHash);

    int* n_array = new int[_rangePow];
    _indices = new int[_rangePow*permute];

    for (int i = 0; i < _rangePow; i++) {
        n_array[i] = i;
    }
    for (int p=0; p<permute ;p++) {
        std::shuffle(n_array, n_array+_rangePow, rd);
        std::copy ( n_array, n_array+_rangePow, _indices+(p*_rangePow) );
    }
    delete [] n_array;
}


int * WtaHash::getHash(float* data)
{

    // binsize is the number of times the range is larger than the total number of hashes we need.

    int *hashes = new int[_numhashes];
    float *values = new float[_numhashes];

    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
        values[i] = INT_MIN;
    }


    for (int i = 0; i < _numhashes; i++)
    {
        for (int j=0; j< binsize; j++){
            if (values[i] < data[_indices[i*binsize+j]]) {
                values[i] = data[i*binsize+j];
                hashes[i] = _indices[i*binsize+j];
            }
        }

    }

    delete[] values;
    return hashes;
}


WtaHash::~WtaHash()
{
}
