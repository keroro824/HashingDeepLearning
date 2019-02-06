#include <iostream>
#include <unordered_map>
#include "LSH.h"
#include <climits>
//#include <ppl.h>
#pragma once

#include "Config.h"
/* Author: Anshumali Shrivastava
*
*/

using namespace std;
//using namespace concurrency;

LSH::LSH(int K, int L)
{
	_K = K;
	_L = L;
	//_range = 1 << 22;
	_rangePow = RANGEROW;
	_bucket = new Bucket*[L];

//#pragma omp parallel for
	for (int i = 0; i < L; i++)
	{
		_bucket[i] = new Bucket[1 << _rangePow]();
	}

	rand1 = new int[_K*_L];

	std::random_device rd;
	 std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, INT_MAX);

//#pragma omp parallel for
	for (int i = 0; i < _K*_L; i++)
	{
		rand1[i] = dis(gen);
		if (rand1[i] % 2 == 0)
			rand1[i]++;
	}
}

void LSH::count()
{
	for (int j=0; j<_L;j++) {
		int total = 0;
		for (int i = 0; i < 1 << _rangePow; i++) {
			cout << _bucket[j][i].getSize() << " ";
			total += _bucket[j][i].getSize();
		}
		cout << endl;
		cout <<"TABLE "<< j << "Total "<< total << endl;
	}

}

int* LSH::hashesToIndex(int * hashes)
{

	int * indices = new int[_L];
	for (int i = 0; i < _L; i++)
	{
		unsigned int index = 0;

		for (int j = 0; j < _K; j++)
		{
			unsigned int h = hashes[_K*i + j];
			h *= rand1[_K*i + j];
			h ^= h >> 13;
			h ^= rand1[_K*i + j];
			index += h * hashes[_K*i + j];
		}

//		index = (index << 2) >> (32 - LSH::_rangePow);
		index = index&((1<<LSH::_rangePow)-1);
		indices[i] = index;
	}

	return indices;
}


int* LSH::add(int *indices, int id)
{
//	int countheavy = 0;
	int * secondIndices = new int[_L];
	for (int i = 0; i < _L; i++)
	{
		secondIndices[i] = _bucket[i][indices[i]].add(id);
	}

	return secondIndices;
}


int LSH::add(int tableId, int indices, int id)
{
//	int countheavy = 0;

	int secondIndices = _bucket[tableId][indices].add(id);

	return secondIndices;
}

/*
* Returns all the buckets
*/
int** LSH::retrieveRaw(int *indices)
{
	int ** rawResults = new int*[_L];
	int count = 0;

	for (int i = 0; i < _L; i++)
	{
		//int *tempArr = _bucket[i][index].getAll();
		//if (_bucket[i][indices[i]].getAll() == NULL)
		//{
		//	continue;
		//}

		rawResults[i] = _bucket[i][indices[i]].getAll();
//        cout<<rawResults[i]<<endl;
	}
	return rawResults;
}


int LSH::retrieve(int table, int indices, int bucket)
{
	return _bucket[table][indices].retrieve(bucket);
}

LSH::~LSH()
{
	delete [] rand1;
	 for (size_t i = 0; i < _L; i++)
	 {
	 	delete[] _bucket[i];
	 }
	 delete[] _bucket;
}

