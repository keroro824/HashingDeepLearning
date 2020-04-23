#pragma once
#include "Layer.h"
#include <vector>

namespace hieu {
class Network {
protected:
  std::vector<Layer *> _layers;

public:
  Network(int);
  virtual ~Network();
};

} // namespace hieu
