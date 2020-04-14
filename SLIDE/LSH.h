#pragma once
#include "Bucket.h"
#include "Util.h"
#include <random>

class LSH {
private:
	Vec2d<Bucket> _bucket;
	int _K;
	int _L;
	int _RangePow;
	std::vector<int> _rand1;


public:
	LSH(int K, int L, int RangePow);
	void clear();
	int* add(int *indices, int id);
	int add(int indices, int tableId, int id);
  std::vector<int> hashesToIndex(const std::vector<int> &hashes);
	int** retrieveRaw(int *indices);
	void count();
	~LSH();
};
