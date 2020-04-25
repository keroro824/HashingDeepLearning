#pragma once
#include "Util.h"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <vector>

enum NodeType { ReLU, Softmax };

struct Train {
  float _lastDeltaforBPs;
  float _lastActivations;
  float _lastGradients;
  bool _ActiveinputIds;
};

class Node {
private:
  int _activeInputs;
  NodeType _type;
  std::vector<Train> _train;
  int _currentBatchsize;
  size_t _layerNum, _IDinLayer;
  size_t _dim;
  SubVector<float> _weights;
  float *_mirrorWeights;

  SubVector<float> _adamAvgMom;
  SubVector<float> _adamAvgVel;
  float _adamAvgMombias = 0;
  float _adamAvgVelbias = 0;

  float _bias = 0;
  float _tbias = 0;
  float _mirrorbias = 0; // not adam

  std::vector<float> _t; // for adam

public:
  const size_t &dim() const { return _dim; }

  const SubVector<float> &weights() const { return _weights; }
  SubVector<float> &weights() { return _weights; }

  const float *mirrorWeights() const { return _mirrorWeights; } // not adam

  SubVector<float> &adamAvgMom() { return _adamAvgMom; }
  SubVector<float> &adamAvgVel() { return _adamAvgVel; }
  float &adamAvgMombias() { return _adamAvgMombias; }
  float &adamAvgVelbias() { return _adamAvgVelbias; }

  float &bias() { return _bias; }
  float &tbias() { return _tbias; }
  const float &mirrorbias() const { return _mirrorbias; }

  float getT(size_t idx) const { return _t[idx]; }
  void setT(size_t idx, float val) { _t[idx] = val; }

  ////////////////////
  Node(){};
  void Update(int dim, int nodeID, int layerID, NodeType type, int batchsize,
              std::vector<float> &allWeights, float bias,
              std::vector<float> &allAdamAvgMom,
              std::vector<float> &allAdamAvgVel);
  float getLastActivation(int inputID) const;
  void incrementDelta(int inputID, float incrementValue);
  float getActivation(const std::vector<int> &indices,
                      const std::vector<float> &values, int length,
                      int inputID);
  void SetlastActivation(int inputID, float realActivation);
  void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID,
                                  const std::vector<int> &label);
  void backPropagate(std::vector<Node> &previousNodes,
                     const std::vector<int> &previousLayerActiveNodeIds,
                     float learningRate, int inputID);
  void backPropagateFirstLayer(const std::vector<int> &nnzindices,
                               const std::vector<float> &nnzvalues,
                               float learningRate, int inputID);
  ~Node();

  // only for debugging
  float purturbWeight(int weightid, float delta);
  float getGradient(int weightid, int inputID, float InputVal);

  const Train &getTrain(size_t idx) const { return _train[idx]; }
};
