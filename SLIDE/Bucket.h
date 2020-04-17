#pragma once
#include "Config.h"
#include <vector>

class Bucket {
private:
  std::vector<int> _arr;
  int _isInit = -1;
  int _index = 0;
  int _counts = 0;

public:
  Bucket();
  void add(int id);
  const std::vector<int> *getAll() const;
  int getSize() const { return _counts; }
  ~Bucket();
};
