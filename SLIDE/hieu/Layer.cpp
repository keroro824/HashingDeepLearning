#include "Layer.h"
#include "../Util.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Layer::Layer(size_t layerIdx, size_t numNodes, size_t prevNumNodes)
  : _layerIdx(layerIdx)
  , _numNodes(numNodes)
  , _prevNumNodes(prevNumNodes)
{

  _weights.resize(numNodes * prevNumNodes);
  _bias.resize(_nodes.size());
  random_device rd;
  default_random_engine dre(rd());
  normal_distribution<float> distribution(0.0, 0.01);

  generate(_weights.begin(), _weights.end(),
           [&]() { return distribution(dre); });
  generate(_bias.begin(), _bias.end(), [&]() { return distribution(dre); });

  _nodes.reserve(numNodes);
  for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
    SubVector<float> nodeWeights =
        SubVector<float>(_weights, nodeIdx * prevNumNodes, prevNumNodes);
    float &nodeBias = _bias[nodeIdx];

    _nodes.emplace_back(new Node(nodeIdx, nodeWeights, nodeBias));
  }

  cerr << "Created Layer"
      << " layerIdx=" << _layerIdx
      << " numNodes=" << _nodes.size() 
      << " prevNumNodes=" << _prevNumNodes
      << endl;
}

Layer::~Layer() {}

size_t Layer::computeActivation(std::vector<float> &dataOut, const std::vector<float> &dataIn) const
{
  assert(dataIn.size() == _prevNumNodes);
  dataOut.resize(_numNodes);
  for (size_t nodeIdx = 0; nodeIdx < _nodes.size(); ++nodeIdx) {
    const Node &node = getNode(nodeIdx);
    dataOut.at(nodeIdx) = node.computeActivation(dataIn);
  }
}


//////////////////////////////////////////
RELULayer::RELULayer(size_t layerIdx, size_t numNodes, size_t prevNumNodes)
    : Layer(layerIdx, numNodes, prevNumNodes) {
  cerr << "Create RELULayer" << endl;
}

RELULayer::~RELULayer() {}

SoftmaxLayer::SoftmaxLayer(size_t layerIdx, size_t numNodes, size_t prevNumNodes)
    : Layer(layerIdx, numNodes, prevNumNodes) {
  cerr << "Create SoftmaxLayer" << endl;
}

SoftmaxLayer::~SoftmaxLayer() {}

} // namespace hieu