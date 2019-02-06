#include <iostream>
#include "Bucket.h"
#pragma once

Bucket::Bucket()
{

    isInit = -1;
//	arr = NULL;
    arr = new int[BUCKETSIZE]();
}

Bucket::~Bucket()
{
//	if (isInit!=-1) {
    delete[] arr;
//	}
}
int Bucket::getTotalCounts()
{
    return _counts;
}

int Bucket::getSize()
{
    return _counts;
}

int Bucket::add(int id)
{


    //Anshu: cyclic adding cyclic

//	if (isInit == -1) {
//		int* tmp = new int[BUCKETSIZE]();
//		isInit = +1;
//		_counts = 0;
//		bool success = __sync_bool_compare_and_swap(&arr, NULL, tmp);
//		if (!success){
//			delete [] tmp;
//		}else{
//			arr = tmp;
//		}
//	}
    isInit +=1;

    int index = _counts& (BUCKETSIZE - 1);
    _counts++;
    arr[index] = id;
    return index;

//		_counts++;
//	#pragma omp critical
//	{
//		if (isInit == -1) {
//			arr = new int[BUCKETSIZE]();
//			isInit = +1;
//		}
//	}
//	if (index == BUCKETSIZE) {
//		//TODO regenerate random number generator for speed
//		int randnum = rand() % (_counts)+1;
//		if (randnum == 2) {
//			int randind = rand() % BUCKETSIZE;
//			arr[randind] = id;
//			return randind;
//		}
//		else
//		{
//			return -1;
//		}
//	}
//	else {
//		arr[index] = id;
//		int returnIndex = index;
//		index++;
//		return returnIndex;
//	}
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
//	for (int i=index; i<BUCKETSIZE; i++){
//		arr[i] = -1;
//	}
//	arr[_counts& (BUCKETSIZE - 1)]=-1;
    if(_counts<BUCKETSIZE){
        arr[_counts]=-1;
    }

//    else if (index>BUCKETSIZE){
//        std::cout<< "Wrong Bucketsize"<<std::endl;
//    }

    return arr;
}
