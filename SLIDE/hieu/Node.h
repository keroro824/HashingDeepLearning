#pragma once
#include "../Util.h"
#include <stddef.h>

namespace hieu {
/////////////////////////////////////////////////////////////
struct Train {
  float _lastDeltaforBPs;
  float _lastActivations;
  float _lastGradients;
  bool _ActiveinputIds = true;
};

/////////////////////////////////////////////////////////////
class Node {
protected:
  size_t _idx;
  SubVector<float> _weights;
  float &_nodeBias;
  std::vector<Train> _train;

  std::vector<float> _t; // for adam
  float *_mirrorWeights = NULL;

  float _bias = 0;
  float _tbias = 0;
  float _mirrorbias = 0; // not adam


  void incrementDelta(int batchIdx, float incrValue);
  float getLastActivation(int batchIdx) const;

public:
  Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias, size_t batchsize);
  virtual ~Node();

  const SubVector<float> &getWeights() const { return _weights; }

  float computeActivation(const std::vector<float> &dataIn) const;

  void backPropagate(std::vector<Node> &prevNodes,
                     const std::vector<int> &prevActiveNodeIdx, float learningRate,
                     size_t batchIdx);
  void backPropagateFirstLayer(const Vec2d<float> &data, float learningRate,
                               size_t batchIdx);
};
} // namespace hieu
