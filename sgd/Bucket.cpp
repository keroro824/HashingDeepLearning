#include <iostream>
#include "Bucket.h"
#include <assert.h>
#include "Config.h"
#pragma once

Bucket::Bucket()
{

	isInit = -1;
	_size=BUCKETSIZE;
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
	return index;
}

int Bucket::add(int id)
{
	_counts++;
	#pragma omp critical
	{
		if (isInit == -1) {
			arr = new int[_size]();
			isInit = +1;
		}
	}
	if (index >= _size) {
		//TODO regenerate random number generator for speed
		int randnum = rand() % (_counts)+1;
		if (randnum == 2) {
			int randind = rand() % _size;
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

int Bucket::retrieve(int index)
{
	assert(("what??", index>=0));
	if (index >= _size)
		return -1;
	return arr[index];
}

int * Bucket::getAll()
{
	if (isInit == -1)
		return NULL;
	for (int i=index; i<_size; i++){
		arr[i] = -1;
	}
	return arr;
}
