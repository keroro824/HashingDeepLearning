#include "Node.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Node::Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias, size_t maxBatchsize)
    : _idx(idx), _nodeWeights(nodeWeights), _nodeBias(nodeBias), _train(maxBatchsize) {
  // cerr << "Create Node" << endl;
}

Node::~Node() {}

float Node::computeActivation(const std::vector<float> &dataIn) const {
  assert(dataIn.size() == _nodeWeights.size());
  float ret = _nodeBias;
  for (size_t idx = 0; idx < _nodeWeights.size(); ++idx) {
    // ret += _nodeWeights[idx] * inVal;
  }

  return ret;
}

void Node::backPropagate(const std::vector<Node> &prevNodes,
                         const std::vector<int> &activeNodesIdx, float tmpLR,
                         size_t batchIdx) {

}

void Node::backPropagateFirstLayer(const Vec2d<float> &data, float tmpLR,
                                   size_t batchIdx) {}

} // namespace hieu
