#pragma once
#include "Bucket.h"
#include "Util.h"
#include <random>
#include <vector>

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
  void add(const std::vector<int> &indices, int id);
	void add(int indices, int tableId, int id);
  std::vector<int> hashesToIndex(const std::vector<int> &hashes) const;
  std::vector<const int*> retrieveRaw(const std::vector<int> &indices);
	void count();
	~LSH();
};
