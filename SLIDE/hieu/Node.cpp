#include "Node.h"
#include <cassert>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Node::Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias, size_t maxBatchsize)
    : _idx(idx), _weights(nodeWeights), _nodeBias(nodeBias), _train(maxBatchsize) {
  // cerr << "Create Node" << endl;
}

Node::~Node() {}

float Node::computeActivation(const std::vector<float> &dataIn) const {
  assert(dataIn.size() == _weights.size());
  float ret = _nodeBias;
  for (size_t idx = 0; idx < _weights.size(); ++idx) {
    // ret += _nodeWeights[idx] * inVal;
  }

  return ret;
}

void Node::backPropagate(std::vector<Node> &prevNodes,
                         const std::vector<int> &prevActiveNodesIdx, float tmpLR,
                         size_t batchIdx) {
  Train &train = _train.at(batchIdx);
  assert(("Input Not Active but still called !! BUG",
    train._ActiveinputIds));
  for (int i = 0; i < prevActiveNodesIdx.size(); i++) {
    // UpdateDelta before updating weights
    int prevActiveNodeIdx = prevActiveNodesIdx.at(i);
    Node &prevNode = prevNodes.at(prevActiveNodeIdx);
    float weight = _weights.at(prevActiveNodeIdx);
    float incrValue = _train.at(batchIdx)._lastDeltaforBPs * weight;
    prevNode.incrementDelta(batchIdx, incrValue);

    float grad_t = train._lastDeltaforBPs * prevNode.getLastActivation(batchIdx);


  }

}

void Node::backPropagateFirstLayer(const Vec2d<float> &data, float tmpLR,
                                   size_t batchIdx) {}

void Node::incrementDelta(int batchIdx, float incrValue)
{
  assert(("Input Not Active but still called !! BUG",
    _train.at(batchIdx)._ActiveinputIds));
  if (_train.at(batchIdx)._lastActivations > 0)
    _train.at(batchIdx)._lastDeltaforBPs += incrValue;

}

float Node::getLastActivation(int batchIdx) const {
  if (!_train.at(batchIdx)._ActiveinputIds)
    return 0.0;
  return _train.at(batchIdx)._lastActivations;
}

} // namespace hieu
