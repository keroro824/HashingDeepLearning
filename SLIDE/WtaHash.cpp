#include "WtaHash.h"
#include "Config.h"
#include <algorithm>
#include <climits>
#include <iostream>
#include <map>
#include <math.h>
#include <random>
#include <vector>
using namespace std;

WtaHash::WtaHash(int numHashes, int noOfBitsToHash) {
  _numhashes = numHashes;
  _rangePow = noOfBitsToHash;

  std::random_device rd;
  std::mt19937 gen(rd());

  int permute = ceil(_numhashes * binsize * 1.0 / noOfBitsToHash);

  std::vector<int> n_array(_rangePow);
  _indices.resize(_rangePow * permute);

  for (int i = 0; i < _rangePow; i++) {
    n_array[i] = i;
  }
  for (int p = 0; p < permute; p++) {
    std::shuffle(n_array.begin(), n_array.end(), rd);
    std::copy(n_array.begin(), n_array.end(),
              _indices.begin() + (p * _rangePow));
  }
}

std::vector<int> WtaHash::getHash(const std::vector<float> &data) const {
  SubVectorConst<float> dataSub(data, 0, data.size());
  return getHash(dataSub);
}

std::vector<int> WtaHash::getHash(const SubVectorConst<float> &data) const {
  // binsize is the number of times the range is larger than the total number of
  // hashes we need.

  std::vector<int> hashes(_numhashes);
  std::vector<float> values(_numhashes);

  for (int i = 0; i < _numhashes; i++) {
    hashes[i] = INT_MIN;
    values[i] = INT_MIN;
  }

  for (int i = 0; i < _numhashes; i++) {
    for (int j = 0; j < binsize; j++) {
      if (values[i] < data[_indices[i * binsize + j]]) {
        values[i] = data[i * binsize + j];
        hashes[i] = _indices[i * binsize + j];
      }
    }
  }

  return hashes;
}

WtaHash::~WtaHash() {}
