#pragma once
#include "MurmurHash.h"
#include "Util.h"
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

class DensifiedMinhash {
private:
  int _randa, _numhashes, _rangePow, _lognumhash;
  std::vector<int> _randHash;

public:
  DensifiedMinhash(int numHashes, int noOfBitsToHash);
  std::vector<int> getHash(const std::vector<int> &indices,
                           const std::vector<float> &data,
                           const std::vector<int> &binids) const;
  int getRandDoubleHash(int binid, int count) const;
  std::vector<int> getHashEasy(const std::vector<int> &binids,
                               const std::vector<float> &data, int topK) const;
  std::vector<int> getHashEasy(const std::vector<int> &binids,
                               const SubVectorConst<float> &data,
                               int topK) const;
  void getMap(int n, std::vector<int> &binid) const;
  ~DensifiedMinhash();
};