#include "WtaHash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
#include <algorithm>
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


void WtaHash::getMap(int n, int* binids, int dim)
{
    int range = 1 << _rangePow;
    // binsize is the number of times the range is larger than the total number of hashes we need.
    int binsize = ceil(1.0*range / _numhashes);

    int bins = ceil(dim/4.0);
//    int binsize;
    if (bins<_numhashes){
        cout<<"Too many hashes"<<endl;
    }else{
        binsize = ceil(1.0*range / bins);
    }

    for (size_t i = 0; i < n; i++)
    {
        unsigned int h = i;
        h *= _randa;
        h ^= h >> 13;
        h *= 0x85ebca6b;
        unsigned int curhash = (unsigned int)(((unsigned int)h*i) << 5);
//        int index = i;
//        uint32_t curhash = MurmurHash( (char *)&index, (uint32_t) sizeof(index), (uint32_t)_randa);
        curhash = curhash & ((1<<_rangePow)-1);
        binids[i] = (int)floor(curhash / binsize);
    }

}


int * WtaHash::getHashEasy(int* binids, float* data, int dataLen, int topK)
{

    // binsize is the number of times the range is larger than the total number of hashes we need.
    //TODO: Sort data, take top sparsity (or 1.5x times) percentage and get the indices, then call the sparse WTA with indices and values
//    vector<pair<float, int> > sortW;
//    vector<pair<float, int> > sortIndex;
//    for (int i=0; i< dataLen; i++){
//        sortW.push_back(make_pair(-olddata[i], oldbinids[i]));
//        sortIndex.push_back(make_pair(-olddata[i], i));
//    }
//    sort(sortW.begin(), sortW.end());
//    sort(sortIndex.begin(), sortIndex.end());
//
////    int* binids=  new int[dataLen];
////    float* data = new float[dataLen];
////    std::copy ( oldbinids, oldbinids+dataLen, binids );
////    std::copy ( olddata, olddata+dataLen, data );
//
////    sort( binids, binids+dataLen, [&](int i,int j){return data[i]>data[j];} );
////    sort( data, data+dataLen, [&](int i,int j){return data[i]>data[j];} );
//
//    dataLen = topK;

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
        int cur = binids[i];
        if(cur<_numhashes) {

            if (values[cur] < data[i]) {
                values[cur] = data[i];
                hashes[cur] = i;
            }
        }
//        if (values[sortW[i].second] < -sortW[i].first) {
//            values[sortW[i].second] = -sortW[i].first;
//            hashes[sortW[i].second] = sortIndex[i].second;
//        }
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
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    delete[] values;
//    delete[] binids;
//    delete[] data;
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

        if(binid<_numhashes) {
            if (values[binid] < data[i]) {
                values[binid] = data[i];
                hashes[binid] = indices[i];
            }
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
            if (count > 100) // Densification failure.
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