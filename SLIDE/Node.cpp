#include "Node.h"
#include "Config.h"
#include <algorithm>
#include <chrono>
#include <math.h>
#include <random>
#include <stdlib.h>
#include <time.h>

using namespace std;

Node::Node() {
  //_mutex = new std::mutex();
}

Node::~Node() {
  //delete _mutex;
}

void Node::Update(int dim, int nodeID, int layerID, NodeType type,
                  int batchsize, std::vector<float> &allWeights, float bias,
                  std::vector<float> &allAdamAvgMom,
                  std::vector<float> &allAdamAvgVel) {
  _dim = dim;
  _IDinLayer = nodeID;
  _type = type;
  _layerNum = layerID;
  _currentBatchsize = batchsize;

  if (ADAM) {
    _adamAvgMom = SubVector<float>(allAdamAvgMom, nodeID * dim, dim);
    _adamAvgVel = SubVector<float>(allAdamAvgVel, nodeID * dim, dim);
    _t.resize(_dim);
  }

  _activeInputs = 0;

  _train.resize(batchsize);
  _weights = SubVector<float>(allWeights, nodeID * dim, dim);
  _bias = bias;
  _mirrorbias = _bias;
}

float Node::getLastActivation(int inputID) const {
  if (!getTrain(inputID)._ActiveinputIds)
    return 0.0;
  return getTrain(inputID)._lastActivations;
}

void Node::incrementDelta(int inputID, float incrementValue) {
  assert(("Input Not Active but still called !! BUG",
    getTrain(inputID)._ActiveinputIds));
  if (getTrain(inputID)._lastActivations > 0)
    getTrain(inputID)._lastDeltaforBPs += incrementValue;
}

float Node::getActivation(const std::vector<int> &indices,
                          const std::vector<float> &values, int length,
                          int inputID) {
  assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));

  // FUTURE TODO: shrink batchsize and check if input is alread active then
  // ignore and ensure backpopagation is ignored too.
  if (!getTrain(inputID)._ActiveinputIds) {
    getTrain(inputID)._ActiveinputIds = true; // activate input
    _activeInputs++;
  }

  getTrain(inputID)._lastActivations = 0;
  for (int i = 0; i < length; i++) {
    getTrain(inputID)._lastActivations += _weights[indices[i]] * values[i];
  }
  getTrain(inputID)._lastActivations += _bias;

  switch (_type) {
  case NodeType::ReLU:
    if (getTrain(inputID)._lastActivations < 0) {
      getTrain(inputID)._lastActivations = 0;
      getTrain(inputID)._lastGradients = 1;
      getTrain(inputID)._lastDeltaforBPs = 0;

    } else {
      getTrain(inputID)._lastGradients = 0;
    }
    break;
  case NodeType::Softmax:

    break;
  default:
    cout << "Invalid Node type from Constructor" << endl;
    break;
  }

  return getTrain(inputID)._lastActivations;
}

void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID,
                                      const std::vector<int> &label) {
  assert(("Input Not Active but still called !! BUG",
          getTrain(inputID)._ActiveinputIds));

  getTrain(inputID)._lastActivations /= normalizationConstant + 0.0000001;

  // TODO:check  gradient
  getTrain(inputID)._lastGradients = 1;
  if (find(label.begin(), label.end(), _IDinLayer) != label.end()) {
    getTrain(inputID)._lastDeltaforBPs =
        (1.0 / label.size() - getTrain(inputID)._lastActivations) /
        _currentBatchsize;
  } else {
    getTrain(inputID)._lastDeltaforBPs =
        (-getTrain(inputID)._lastActivations) / _currentBatchsize;
  }
}

void Node::backPropagate(std::vector<Node> &previousNodes,
                         const std::vector<int> &previousLayerActiveNodeIds,
                         float learningRate,
                         int inputID) {
  assert(("Input Not Active but still called !! BUG",
          getTrain(inputID)._ActiveinputIds));
  for (int i = 0; i < previousLayerActiveNodeIds.size(); i++) {
    // UpdateDelta before updating weights
    Node &prev_node = previousNodes[previousLayerActiveNodeIds[i]];
    prev_node.incrementDelta(inputID,
                             getTrain(inputID)._lastDeltaforBPs *
                                 _weights[previousLayerActiveNodeIds[i]]);

    float grad_t =
        getTrain(inputID)._lastDeltaforBPs * prev_node.getLastActivation(inputID);

    if (ADAM) {
      _t[previousLayerActiveNodeIds[i]] += grad_t;
    } else {
      _mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
    }
  }

  if (ADAM) {
    float biasgrad_t = getTrain(inputID)._lastDeltaforBPs;
    float biasgrad_tsq = biasgrad_t * biasgrad_t;
    _tbias += biasgrad_t;
  } else {
    _mirrorbias += learningRate * getTrain(inputID)._lastDeltaforBPs;
  }

  getTrain(inputID)._ActiveinputIds = false;
  getTrain(inputID)._lastDeltaforBPs = 0;
  getTrain(inputID)._lastActivations = 0;
  _activeInputs--;
}

void Node::backPropagateFirstLayer(const std::vector<int> &nnzindices,
                                   const std::vector<float> &nnzvalues,
                                   float learningRate, int inputID) {
  assert(("Input Not Active but still called !! BUG",
          getTrain(inputID)._ActiveinputIds));
  for (int i = 0; i < nnzindices.size(); i++) {
    float grad_t = getTrain(inputID)._lastDeltaforBPs * nnzvalues[i];
    float grad_tsq = grad_t * grad_t;
    if (ADAM) {
      _t[nnzindices[i]] += grad_t;
    } else {
      _mirrorWeights[nnzindices[i]] += learningRate * grad_t;
    }
  }

  if (ADAM) {
    float biasgrad_t = getTrain(inputID)._lastDeltaforBPs;
    float biasgrad_tsq = biasgrad_t * biasgrad_t;
    _tbias += biasgrad_t;
  } else {
    _mirrorbias += learningRate * getTrain(inputID)._lastDeltaforBPs;
  }

  getTrain(inputID)._ActiveinputIds = false; // deactivate inputIDs
  getTrain(inputID)._lastDeltaforBPs = 0;
  getTrain(inputID)._lastActivations = 0;
  _activeInputs--;
}

void Node::SetlastActivation(int inputID, float realActivation) {
  getTrain(inputID)._lastActivations = realActivation;
}

// for debugging gradients.
float Node::purturbWeight(int weightid, float delta) {
  _weights[weightid] += delta;
  return _weights[weightid];
}

float Node::getGradient(int weightid, int inputID, float InputVal) {
  return -getTrain(inputID)._lastDeltaforBPs * InputVal;
}

const Train &Node::getTrain(size_t idx) const { 
  //std::lock_guard<std::mutex> guard(*_mutex);
  return _train[idx];
  /*
  BatchTrain::const_iterator iter = _train.find(idx);
  assert(iter != _train.end());
  return iter->second;
  */
}

Train &Node::getTrain(size_t idx) {
  //std::lock_guard<std::mutex> guard(*_mutex);
  return _train[idx];
}
