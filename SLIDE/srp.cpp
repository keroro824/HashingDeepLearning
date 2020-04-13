#include "srp.h"
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

SparseRandomProjection::SparseRandomProjection(size_t dimension, size_t numOfHashes, int ratio) {
    _dim = dimension;
    _numhashes = numOfHashes;
    _samSize = ceil(1.0*_dim / ratio);

    int *a = new int[_dim];
    for (size_t i = 0; i < _dim; i++) {
        a[i] = i;
    }

    srand(time(0));
    _randBits = new short *[_numhashes];
    _indices = new int *[_numhashes];

    for (size_t i = 0; i < _numhashes; i++) {
        random_shuffle(a, a+_dim);
        _randBits[i] = new short[_samSize];
        _indices[i] = new int[_samSize];
        for (size_t j = 0; j < _samSize; j++) {
            _indices[i][j] = a[j];
            int curr = rand();
            if (curr % 2 == 0) {
                _randBits[i][j] = 1;
            } else {
                _randBits[i][j] = -1;
            }
        }
        std::sort(_indices[i], _indices[i]+_samSize);
    }
    delete [] a;
}


std::vector<int> SparseRandomProjection::getHash(const std::vector<float> &vector, int length)
{
  SubVectorConst<float> dataSub(vector, 0, vector.size());
  return getHash(vector, length);
}

std::vector<int> SparseRandomProjection::getHash(const SubVectorConst<float> &vector, int length)
{
    // length should be = to _dim
    std::vector<int> hashes(_numhashes);

 // #pragma omp parallel for
    for (size_t i = 0; i < _numhashes; i++) {
        double s = 0;
        for (size_t j = 0; j < _samSize; j++) {
            float v = vector[_indices[i][j]];
            if (_randBits[i][j] >= 0) {
                s += v;
            } else {
                s -= v;
            }
        }
        hashes[i] = (s >= 0 ? 0 : 1);
    }
    return hashes;
}


std::vector<int> SparseRandomProjection::getHashSparse(const std::vector<int> &indices, const std::vector<float> &values, size_t length) {
  std::vector<int> hashes(_numhashes);

    for (size_t p = 0; p < _numhashes; p++) {
        double s = 0;
        size_t i = 0;
        size_t j = 0;
        while (i < length & j < _samSize) {
            if (indices[i] == _indices[p][j]) {
                float v = values[i];
                if (_randBits[p][j] >= 0) {
                    s += v;
                } else {
                    s -= v;
                }
                i++;
                j++;
            }
            else if (indices[i] < _indices[p][j]){
                i++;
            }
            else{
                j++;
            }
        }
        hashes[p] = (s >= 0 ? 0 : 1);
    }

    return hashes;
}


SparseRandomProjection::~SparseRandomProjection() {
    for (size_t i = 0; i < _numhashes; i++) {
        delete[]   _randBits[i];
        delete[]   _indices[i];
    }
    delete[]   _randBits;
    delete[]   _indices;
}
