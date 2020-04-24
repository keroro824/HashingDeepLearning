#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Network::Network(size_t maxBatchsize) {
  size_t inputDim = 135909;

  cerr << "Create Network" << endl;
  _layers.push_back(new RELULayer(0, 128, inputDim, maxBatchsize));
  _layers.push_back(new SoftmaxLayer(1, 670091, 128, maxBatchsize));
}

Network::~Network() { cerr << "~Network" << endl; }

size_t Network::predictClass(const Vec2d<float> &data,
                             const Vec2d<int> &labels) const {
  assert(data.size() == labels.size());
  size_t batchSize = data.size();

  size_t correctPred = 0;

  // inference
  for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
    const std::vector<float> &data1 = data.at(batchIdx);
    const std::vector<int> &labels1 = labels.at(batchIdx);

    const std::vector<float> *lastActivations =
        computeActivation(data1, labels1);

    delete lastActivations;
  }
  return correctPred;
}

const std::vector<float> *
Network::computeActivation(const std::vector<float> &data1,
                           const std::vector<int> &labels1) const {
  size_t correctPred = 0;

  std::vector<float> *dataOut = new std::vector<float>;

  const Layer &firstLayer = getLayer(0);
  firstLayer.computeActivation(*dataOut, data1);

  std::vector<float> *dataIn = dataOut;
  dataOut = new std::vector<float>;

  for (int layerIdx = 1; layerIdx < _layers.size(); ++layerIdx) {
    const Layer &layer = getLayer(layerIdx);
    layer.computeActivation(*dataOut, *dataIn);

    std::swap(dataIn, dataOut);
  }

  delete dataOut;

  return dataIn;
}

float Network::ProcessInput(const Vec2d<float> &data, const Vec2d<int> &labels,
                            int iter, bool rehash, bool rebuild) {
  assert(data.size() == labels.size());
  size_t batchSize = data.size();

  float logloss = 0.0;
  float tmpLR = 24534534.5;

  for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
    const std::vector<float> &data1 = data.at(batchIdx);
    const std::vector<int> &labels1 = labels.at(batchIdx);

    const std::vector<float> *lastActivations =
        computeActivation(data1, labels1);

    delete lastActivations;

    // Now backpropagate.
    for (int layerIdx = _layers.size() - 1; layerIdx >= 0; layerIdx--) {
      Layer &layer = getLayer(layerIdx);

      // active nodes
      std::vector<int> activeNodesIdx(layer.getNumNodes());
      for (size_t nodeIdx = 0; nodeIdx < layer.getNumNodes(); ++nodeIdx) {
        activeNodesIdx[nodeIdx] = nodeIdx;
      }

      cerr << "batchIdx=" << batchIdx << " layerIdx=" << layerIdx << endl;

      for (size_t nodeIdx : activeNodesIdx) {
        Node &node = layer.getNode(nodeIdx);

        if (layerIdx > 0) {
          Layer &prevLayer = getLayer(layerIdx - 1);
          node.backPropagate(prevLayer.getNodes(), activeNodesIdx, tmpLR,
                             batchIdx);
        } else {
          node.backPropagateFirstLayer(data, tmpLR, batchIdx);
        }
      }
    }
  }
}

} // namespace hieu