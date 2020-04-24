#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Network::Network() {
  size_t inputDim = 135909;

  cerr << "Create Network" << endl;
  _layers.push_back(new RELULayer(0, 128, inputDim));
  _layers.push_back(new SoftmaxLayer(1, 670091, 128));
}

Network::~Network() { cerr << "~Network" << endl; }

size_t
Network::predictClass(const Vec2d<float> &data,
                      const Vec2d<int> &labels) const {
  assert(data.size() == labels.size());
  size_t batchSize = data.size();

  size_t correctPred = 0;

  // inference
  for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
    const std::vector<float> &data1 = data.at(batchIdx);
    const std::vector<int> &labels1 = labels.at(batchIdx);

    const std::vector<float> *lastActivations = computeActivation(data1, labels1);
  }
  return correctPred;
}

const std::vector<float> *Network::computeActivation(const std::vector<float> &data1,
                                  const std::vector<int> &labels1) const {
  size_t correctPred = 0;

  std::vector<float> *dataOut = new std::vector<float>;

  cerr << "layerIdx0" << endl;
  const Layer &firstLayer = getLayer(0);
  firstLayer.computeActivation(*dataOut, data1);

  std::vector<float> *dataIn = dataOut;
  dataOut = new std::vector<float>;

  for (int layerIdx = 1; layerIdx < _layers.size(); ++layerIdx) {
    cerr << "layerIdx=" << layerIdx << endl;
    cerr << "dataIn=" << dataIn->size() << endl;
    const Layer &layer = getLayer(layerIdx);
    layer.computeActivation(*dataOut, *dataIn);
    cerr << "dataOut=" << dataOut->size() << endl;

    std::swap(dataIn, dataOut);
  }

  cerr << "computeActivation1" << endl;
  delete dataOut;
  cerr << "computeActivation2" << endl;

  return dataIn;
}

} // namespace hieu