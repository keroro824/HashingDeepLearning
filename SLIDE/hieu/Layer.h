#pragma once
#include "Node.h"
#include <vector>
#include <unordered_map>

namespace hieu {
class Layer {
protected:
  std::vector<Node> _nodes;
  std::vector<float> _weights;
  std::vector<float> _bias;

public:
  Layer(size_t numNodes, size_t prevNumNodes);
  virtual ~Layer();

  virtual size_t computeActivation(std::vector<float> &dataOut, const std::vector<float> &dataIn) const;
};

class RELULayer : public Layer {
protected:
public:
  RELULayer(size_t numNodes, size_t prevNumNodes);
  virtual ~RELULayer();
};

class SoftmaxLayer : public Layer {
protected:
public:
  SoftmaxLayer(size_t numNodes, size_t prevNumNodes);
  virtual ~SoftmaxLayer();
};

} // namespace hieu
