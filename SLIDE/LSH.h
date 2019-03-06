#pragma once
#include "Bucket.h"
#include <random>

class LSH {
private:
	Bucket ** _bucket;
	int _K;
	int _L;
	int _RangePow;
	int *rand1;


public:
	LSH(int K, int L, int RangePow);
	void clear();
	int* add(int *indices, int id);
	int add(int indices, int tableId, int id);
	int * hashesToIndex(int * hashes);
	int** retrieveRaw(int *indices);
	int retrieve(int table, int indices, int bucket);
	void count();
	~LSH();
};
