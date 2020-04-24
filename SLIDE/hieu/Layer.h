#pragma once
#include "Node.h"
#include <unordered_map>
#include <vector>

namespace hieu {
class Layer {
protected:
  std::vector<Node> _nodes;
  std::vector<float> _weights;
  std::vector<float> _bias;
  size_t _layerIdx, _numNodes, _prevNumNodes;

public:
  Layer(size_t layerIdx, size_t numNodes, size_t prevNumNodes);
  virtual ~Layer();

  virtual size_t computeActivation(std::vector<float> &dataOut,
                                   const std::vector<float> &dataIn) const;

  size_t getNumNodes() const { return _nodes.size(); }
  const Node &getNode(size_t idx) const { return _nodes.at(idx); }
  Node &getNode(size_t idx) { return _nodes.at(idx); }
  const std::vector<Node> &getNodes() const { return _nodes; }

};

class RELULayer : public Layer {
protected:
public:
  RELULayer(size_t layerIdx, size_t numNodes, size_t prevNumNodes);
  virtual ~RELULayer();
};

class SoftmaxLayer : public Layer {
protected:
public:
  SoftmaxLayer(size_t layerIdx, size_t numNodes, size_t prevNumNodes);
  virtual ~SoftmaxLayer();
};

} // namespace hieu
