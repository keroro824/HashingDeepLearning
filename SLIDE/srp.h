#include <vector>
#pragma once
using namespace std;

class SparseRandomProjection 
{
private:
	size_t _dim;
	size_t _numhashes, _samSize;
	short ** _randBits;
	int ** _indices;
public:
	SparseRandomProjection(size_t dimention, size_t numOfHashes, int ratio);
	int * getHash(const float * vector, int length);
	int * getHashSparse(const int* indices, const float *values, size_t length);
	~SparseRandomProjection();
};
