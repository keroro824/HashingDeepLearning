#include "Layer.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
  Layer::Layer(size_t numNodes) {
    for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
      _nodes.emplace_back(Node(nodeIdx));
    }

    cerr << "Created Layer, numNodes=" << _nodes.size() << endl;
  }

  Layer::~Layer() {

  }

//////////////////////////////////////////
  RELULayer::RELULayer(size_t numNodes)
  :Layer(numNodes) {
    cerr << "Create RELULayer" << endl;
  }

  RELULayer::~RELULayer() {
  }

  SoftmaxLayer::SoftmaxLayer(size_t numNodes)
    :Layer(numNodes) {      
    cerr << "Create SoftmaxLayer" << endl;

  }

  SoftmaxLayer::~SoftmaxLayer()
  {}





}