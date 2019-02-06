#include <iostream>
#include "Bucket.h"
#pragma once

Bucket::Bucket()
{

	isInit = -1;
}

Bucket::~Bucket()
{
	if (isInit!=-1) {
		delete[] arr;
	}
}
int Bucket::getTotalCounts()
{
	return _counts;
}

int Bucket::getSize()
{
	return index;
}

int Bucket::add(int id)
{
	_counts++;
	#pragma omp critical
	{
		if (isInit == -1) {
			arr = new int[BUCKETSIZE]();
			isInit = +1;
		}
	}
	if (index == BUCKETSIZE) {
		//TODO regenerate random number generator for speed
		int randnum = rand() % (_counts)+1;
		if (randnum == 2) {
			int randind = rand() % BUCKETSIZE;
			arr[randind] = id;
			return randind;
		}
		else
		{
			return -1;
		}
	}
	else {
		arr[index] = id;
		int returnIndex = index;
		index++;
		return returnIndex;
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
//	for (int i=index; i<BUCKETSIZE; i++){
//		arr[i] = -1;
//	}
	arr[index]=-1;
	return arr;
}
