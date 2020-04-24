#pragma once
#include "Layer.h"
#include <unordered_map>
#include <vector>

namespace hieu {
class Network {
protected:
  std::vector<Layer *> _layers;

  const Layer &getLayer(size_t idx) const { return *_layers.at(idx); }
  Layer &getLayer(size_t idx) { return *_layers.at(idx); }

  const std::vector<float> *
  computeActivation(const std::vector<float> &data1,
                    const std::vector<int> &labels) const;

public:
  Network(size_t maxBatchsize);
  virtual ~Network();

  size_t predictClass(const Vec2d<float> &data, const Vec2d<int> &labels) const;

  float ProcessInput(const Vec2d<float> &data, const Vec2d<int> &labels,
                     int iter, bool rehash, bool rebuild);
};

} // namespace hieu
