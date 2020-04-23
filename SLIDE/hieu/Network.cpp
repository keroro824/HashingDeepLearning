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

size_t
Network::predictClass(const std::vector<std::unordered_map<int, float>> &data,
                      const Vec2d<int> &labels) const {
  assert(data.size() == labels.size());
  size_t batchSize = data.size();

  size_t correctPred = 0;

  // inference
  for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
    const std::unordered_map<int, float> &data1 = data[batchIdx];
    const std::vector<int> &labels1 = labels[batchIdx];

    size_t correctPred1 = computeActivation(data1, labels1);
  }
  return correctPred;
}

size_t Network::computeActivation(const std::unordered_map<int, float> &data1,
                                  const std::vector<int> &labels1) const {
  size_t correctPred = 0;

  // inference
  const Layer &layer = getLayer(0);

  for (int layerIdx = 1; layerIdx < _layers.size(); ++layerIdx) {
    const Layer &layer = getLayer(layerIdx);
  }

  return correctPred;
}

} // namespace hieu