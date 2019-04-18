#include <iostream>
#include "Bucket.h"


Bucket::Bucket()
{
    isInit = -1;
    arr = new int[BUCKETSIZE]();
}


Bucket::~Bucket()
{
    delete[] arr;
}


int Bucket::getTotalCounts()
{
    return _counts;
}


int Bucket::getSize()
{
    return _counts;
}


int Bucket::add(int id) {

    //FIFO
    if (FIFO) {
        isInit += 1;
        int index = _counts & (BUCKETSIZE - 1);
        _counts++;
        arr[index] = id;
        return index;
    }
    //Reservoir Sampling
    else {
        _counts++;
        if (index == BUCKETSIZE) {
            int randnum = rand() % (_counts) + 1;
            if (randnum == 2) {
                int randind = rand() % BUCKETSIZE;
                arr[randind] = id;
                return randind;
            } else {
                return -1;
            }
        } else {
            arr[index] = id;
            int returnIndex = index;
            index++;
            return returnIndex;
        }
    }
}


int Bucket::retrieve(int indice)
{
    if (indice >= BUCKETSIZE)
        return -1;
    return arr[indice];
}


int * Bucket::getAll()
{
    if (isInit == -1)
        return NULL;
    if(_counts<BUCKETSIZE){
        arr[_counts]=-1;
    }
    return arr;
}
