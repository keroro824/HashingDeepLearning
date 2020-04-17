#include "LSH.h"
#include "Config.h"
#include <chrono>
#include <climits>
#include <iostream>
#include <unordered_map>

using namespace std;

LSH::LSH(int K, int L, int RangePow) : _rand1(K * L), _bucket(L) {
  _K = K;
  _L = L;
  _RangePow = RangePow;

  //#pragma omp parallel for
  for (int i = 0; i < L; i++) {
    _bucket[i].resize(1 << _RangePow);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, INT_MAX);

  //#pragma omp parallel for
  for (int i = 0; i < _K * _L; i++) {
    _rand1[i] = dis(gen);
    if (_rand1[i] % 2 == 0)
      _rand1[i]++;
  }
}

void LSH::clear() {
  for (int i = 0; i < _L; i++) {
    _bucket[i].clear();
    _bucket[i].resize(1 << _RangePow);
  }
}

void LSH::count() {
  for (int j = 0; j < _L; j++) {
    int total = 0;
    for (int i = 0; i < 1 << _RangePow; i++) {
      if (_bucket[j][i].getSize() != 0) {
        cout << _bucket[j][i].getSize() << " ";
      }
      total += _bucket[j][i].getSize();
    }
    cout << endl;
    cout << "TABLE " << j << "Total " << total << endl;
  }
}

std::vector<int> LSH::hashesToIndex(const std::vector<int> &hashes) const {
  std::vector<int> indices(_L);
  for (int i = 0; i < _L; i++) {
    unsigned int index = 0;

    for (int j = 0; j < _K; j++) {
      if (HashFunction == 4) {
        unsigned int h = hashes[_K * i + j];
        index += h << (_K - 1 - j);
      } else if (HashFunction == 1 | HashFunction == 2) {
        unsigned int h = hashes[_K * i + j];
        index += h << ((_K - 1 - j) * (int)floor(log(binsize)));

      } else {
        unsigned int h = _rand1[_K * i + j];
        h *= _rand1[_K * i + j];
        h ^= h >> 13;
        h ^= _rand1[_K * i + j];
        index += h * hashes[_K * i + j];
      }
    }
    if (HashFunction == 3) {
      index = index & ((1 << _RangePow) - 1);
    }
    indices[i] = index;
  }

  return indices;
}

void LSH::add(const std::vector<int> &indices, int id) {
  for (int i = 0; i < _L; i++) {
    _bucket[i][indices[i]].add(id);
  }
}

void LSH::add(int tableId, int indices, int id) {
  _bucket[tableId][indices].add(id);
}

/*
 * Returns all the buckets
 */
std::vector<const std::vector<int>*> LSH::retrieveRaw(const std::vector<int> &indices) {
  std::vector<const std::vector<int>*> rawResults(_L);

  for (int i = 0; i < _L; i++) {
    rawResults[i] = _bucket[i][indices[i]].getAll();
  }
  return rawResults;
}

LSH::~LSH() {}
