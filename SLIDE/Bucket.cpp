#include "Bucket.h"
#include "Util.h"
#include <iostream>

using namespace std;

Bucket::Bucket() { _arr.reserve(BUCKETSIZE); }

Bucket::~Bucket() {}

void Bucket::add(int id) {
  if (_arr.size() < BUCKETSIZE) {
    _arr.push_back(id);
  } else {
    // FIFO
    if (FIFO) {
      int index = _counts & (BUCKETSIZE - 1);
      _arr.at(index) = id;
      // return index;
    }
    // Reservoir Sampling
    else {
      /*
      _counts++;
      if (_index == BUCKETSIZE) {
        int randnum = rand() % (_counts) + 1;
        if (randnum == 2) {
          int randind = rand() % BUCKETSIZE;
          _arr.at(randind) = id;
          // return randind;
        } else {
          // return -1;
        }
      } else {
        _arr.at(_index) = id;
        int returnIndex = _index;
        _index++;
        // return returnIndex;
      }
    */
    }
  }
  ++_counts;
}

const std::vector<int> &Bucket::getAll() const { return _arr; }
