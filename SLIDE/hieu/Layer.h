#pragma once
#include "Node.h"
#include <vector>

namespace hieu {
  class Layer {
  protected:
    std::vector<Node> _nodes;
  public:
    Layer();
    virtual ~Layer();

  };
}
