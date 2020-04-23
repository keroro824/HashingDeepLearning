#pragma once
#include "Layer.h"
#include <vector>

namespace hieu {
class Network {
protected:
  std::vector<Layer *> _layers;

  const Layer &getLayer(size_t idx) const { return *_layers[idx]; }
  Layer &getLayer(size_t idx) { return *_layers[idx]; }

  size_t computeActivation(const std::vector<int> &inputIndices, const std::vector<float> &inputValues,
    const std::vector<int> &labels) const;

public:
  Network();
  virtual ~Network();

  size_t predictClass(const Vec2d<int> &inputIndices, const Vec2d<float> &inputValues,
    const Vec2d<int> &labels) const;

};

} // namespace hieu
