#include <vector>
#pragma once
using namespace std;

class SparseRandomProjection 
{
private:
	int _dim;
	int _numhashes, _samSize;
	short ** _randBits;
	int ** _indices;
public:
	SparseRandomProjection(int dimention, int numOfHashes, int ratio);
	int * getHash(float * vector, int length);
	int * getHashSparse(int* indices, float *values, int length);
	~SparseRandomProjection();
};