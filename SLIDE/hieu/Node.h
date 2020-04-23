#pragma once
#include <stddef.h>
#include "../Util.h"

namespace hieu {
  class Node {
  protected:
    size_t _idx;
    SubVector<float> &_nodeWeights;
    float &_nodeBias;

  public:
    Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias);
    virtual ~Node();

  };
}
