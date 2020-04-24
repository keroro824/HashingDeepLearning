#pragma once
#include "../Util.h"
#include <stddef.h>

namespace hieu {
class Node {
protected:
  size_t _idx;
  SubVector<float> _nodeWeights;
  float &_nodeBias;

public:
  Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias);
  virtual ~Node();

  const SubVector<float> &getWeights() const { return _nodeWeights; }

  float computeActivation(const std::vector<float> &dataIn) const;
  
  void backPropagate(const std::vector<Node> &prevNodes, const std::vector<int> &activeNodesIdx, float tmpLR, size_t batchIdx);
  void backPropagateFirstLayer(const Vec2d<float> &data, float tmpLR, size_t batchIdx);

};
} // namespace hieu
