#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Network::Network() {
  size_t inputDim = 135909;

  cerr << "Create Network" << endl;
  _layers.push_back(new RELULayer(128, inputDim));
  _layers.push_back(new SoftmaxLayer(670091, 128));
}

Network::~Network() { cerr << "~Network" << endl; }

size_t Network::predictClass(const Vec2d<int> &inputIndices, const Vec2d<float> &inputValues,
  const Vec2d<int> &labels) const {
  assert(inputIndices.size() == inputValues.size());
  assert(inputIndices.size() == labels.size());
  size_t batchSize = inputIndices.size();

  size_t correctPred = 0;

  // inference
  for (int batchSize = 0; batchSize < batchSize; ++batchSize) {
    const std::vector<int> &inputIndices1 = inputIndices[batchSize];
    const std::vector<float> &inputValues1 = inputValues[batchSize];
    const std::vector<int> &labels1 = labels[batchSize];
    
    size_t correctPred1 = computeActivation(inputIndices1, inputValues1, labels1);

  }


  return correctPred;
}

size_t Network::computeActivation(const std::vector<int> &inputIndices1, const std::vector<float> &inputValues1,
  const std::vector<int> &labels1) const {
  assert(inputIndices1.size() == inputValues1.size());

  size_t correctPred = 0;

  // inference
  const Layer &layer = getLayer(0);

  for (int layerIdx = 1; layerIdx < _layers.size(); ++layerIdx) {
    const Layer &layer = getLayer(layerIdx);

  }


  return correctPred;
}

} // namespace hieu