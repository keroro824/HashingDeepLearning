#include <iostream>
#include "Bucket.h"

using namespace std;

Bucket::Bucket()
:isInit(-1)
,arr(BUCKETSIZE)
{
}


Bucket::~Bucket()
{
}

void Bucket::add(int id) {

    //FIFO
    if (FIFO) {
        isInit += 1;
        int index = _counts & (BUCKETSIZE - 1);
        _counts++;
        arr.at(index) = id;
        //return index;
    }
    //Reservoir Sampling
    else {
        _counts++;
        if (index == BUCKETSIZE) {
            int randnum = rand() % (_counts) + 1;
            if (randnum == 2) {
                int randind = rand() % BUCKETSIZE;
                arr.at(randind) = id;
                //return randind;
            } else {
                //return -1;
            }
        } else {
            arr.at(index) = id;
            int returnIndex = index;
            index++;
            //return returnIndex;
        }
    }
}

int * Bucket::getAll()
{
    if (isInit == -1)
        return NULL;
    if(_counts<BUCKETSIZE){
        arr.at(_counts)=-1;
    }
    return arr.data();
}
