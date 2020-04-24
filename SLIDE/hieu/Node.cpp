#include "Node.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Node::Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias)
    : _idx(idx), _nodeWeights(nodeWeights), _nodeBias(nodeBias) {
  // cerr << "Create Node" << endl;
}

Node::~Node() {}

float Node::computeActivation(const std::vector<float> &dataIn) const {
  assert(dataIn.size() == _nodeWeights.size());
  float ret = 0;
  for (size_t idx = 0; idx < _nodeWeights.size(); ++idx) {
    // ret += _nodeWeights[idx] * inVal;
  }

  return ret;
}

} // namespace hieu
