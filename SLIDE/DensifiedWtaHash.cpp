#include "DensifiedWtaHash.h"
#include "Config.h"
#include "Util.h"
#include <algorithm>
#include <climits>
#include <iostream>
#include <map>
#include <math.h>
#include <random>
#include <vector>

using namespace std;

DensifiedWtaHash::DensifiedWtaHash(int numHashes, int noOfBitsToHash)
    : _randHash(2) {
  _numhashes = numHashes;
  _rangePow = noOfBitsToHash;

  std::random_device rd;
  std::mt19937 gen(rd());

  _permute = ceil(_numhashes * binsize * 1.0 / noOfBitsToHash);

  std::vector<int> n_array(_rangePow);
  _indices.resize(_rangePow * _permute);
  _pos.resize(_rangePow * _permute);

  cerr << "numHashes=" << numHashes << endl;
  cerr << "noOfBitsToHash=_rangePow=" << noOfBitsToHash << endl;
  cerr << "_permute=" << _permute << endl;
  cerr << "_indices.size()=" << _indices.size() << endl;
  cerr << "_pos.size()=" << _pos.size() << endl;

  for (int i = 0; i < _rangePow; i++) {
    n_array[i] = i;
  }

  for (int p = 0; p < _permute; p++) {
    std::shuffle(n_array.begin(), n_array.end(), rd);
    for (int j = 0; j < _rangePow; j++) {
      _indices[p * _rangePow + n_array[j]] = (p * _rangePow + j) / binsize;
      _pos[p * _rangePow + n_array[j]] = (p * _rangePow + j) % binsize;
    }
  }

  _lognumhash = log2(numHashes);
  std::uniform_int_distribution<> dis(1, INT_MAX);

  _randa = dis(gen);
  if (_randa % 2 == 0)
    _randa++;

  _randHash[0] = dis(gen);
  if (_randHash[0] % 2 == 0)
    _randHash[0]++;
  _randHash[1] = dis(gen);
  if (_randHash[1] % 2 == 0)
    _randHash[1]++;
}

std::vector<int> DensifiedWtaHash::getHashEasy(const std::vector<float> &data,
                                               int topK) const {
  SubVectorConst<float> dataSub(data, 0, data.size());
  return getHashEasy(dataSub, topK);
}

std::vector<int>
DensifiedWtaHash::getHashEasy(const SubVectorConst<float> &data,
                              int topK) const {
  // binsize is the number of times the range is larger than the total number of
  // hashes we need.
  std::vector<int> hashes(_numhashes);
  std::vector<float> values(_numhashes);
  std::vector<int> hashArray(_numhashes);

  for (int i = 0; i < _numhashes; i++) {
    hashes[i] = INT_MIN;
    values[i] = INT_MIN;
  }

  for (int p = 0; p < _permute; p++) {
    int bin_index = p * _rangePow;
    for (int i = 0; i < data.size(); i++) {
      int inner_index = bin_index + i;
      int binid = _indices[inner_index];
      float loc_data = data[i];
      if (binid < _numhashes && values[binid] < loc_data) {
        values[binid] = loc_data;
        hashes[binid] = _pos[inner_index];
      }
    }
  }

  for (int i = 0; i < _numhashes; i++) {
    int next = hashes[i];
    if (next != INT_MIN) {
      hashArray[i] = hashes[i];
      continue;
    }
    int count = 0;
    while (next == INT_MIN) {
      count++;
      int index = std::min(getRandDoubleHash(i, count), _numhashes);

      next = hashes[index]; // Kills GPU.
      if (count > 100)      // Densification failure.
        break;
    }
    hashArray[i] = next;
  }

  //cerr << "data:" << data.size() << endl;
  //Print(data);

  //cerr << "hashArray:" << hashArray.size() << endl;;
  //Print(hashArray);

  return hashArray;
}

std::vector<int>
DensifiedWtaHash::getHash(const std::vector<int> &indices,
                          const std::vector<float> &data) const {
  std::vector<int> hashes(_numhashes);
  std::vector<float> values(_numhashes);
  std::vector<int> hashArray(_numhashes);

  // init hashes and values to INT_MIN to start
  for (int i = 0; i < _numhashes; i++) {
    hashes[i] = INT_MIN;
    values[i] = INT_MIN;
  }

  //
  for (int p = 0; p < _permute; p++) {
    for (int i = 0; i < data.size(); i++) {
      int binid = _indices[p * _rangePow + indices[i]];
      if (binid < _numhashes) {
        if (values[binid] < data[i]) {
          values[binid] = data[i];
          hashes[binid] = _pos[p * _rangePow + indices[i]];
        }
      }
    }
  }

  for (int i = 0; i < _numhashes; i++) {
    int next = hashes[i];
    if (next != INT_MIN) {
      hashArray[i] = hashes[i];
      continue;
    }
    int count = 0;
    while (next == INT_MIN) {
      count++;
      int index = std::min(getRandDoubleHash(i, count), _numhashes);

      next = hashes[index]; // Kills GPU.
      if (count > 100)      // Densification failure.
        break;
    }
    hashArray[i] = next;
  }

  return hashArray;
}

int DensifiedWtaHash::getRandDoubleHash(int binid, int count) const {
  unsigned int tohash = ((binid + 1) << 6) + count;
  return (_randHash[0] * tohash << 3) >>
         (32 - _lognumhash); // _lognumhash needs to be ceiled.
}

DensifiedWtaHash::~DensifiedWtaHash() {}
