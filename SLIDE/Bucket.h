#pragma once
#include "Config.h"
#include <mutex>
#include <vector>

class Bucket {
private:
  std::vector<int> _arr;
  int _counts = 0;
  std::mutex *_mutex;

public:
  Bucket();
  virtual ~Bucket();

  void add(int id);
  const std::vector<int> &getAll() const;
};
