#pragma once
#include "Node.h"
#include <vector>

namespace hieu {
  class Layer {
  protected:
    std::vector<Node> _nodes;
  public:
    Layer(size_t numNodes);
    virtual ~Layer();

  };

  class RELULayer: public Layer {
  protected:
  public:
    RELULayer(size_t numNodes);
    virtual ~RELULayer();

  };

  class SoftmaxLayer : public Layer {
  protected:
  public:
    SoftmaxLayer(size_t numNodes);
    virtual ~SoftmaxLayer();

  };


}
