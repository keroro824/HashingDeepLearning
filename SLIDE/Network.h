#pragma once
#include "Layer.h"
#include <chrono>
#include <vector>
#include "cnpy.h"
#include <sys/mman.h>

class Network
{
private:
	std::vector<Layer*> _hiddenlayers;
	const float _learningRate;
  const int _numberOfLayers;
  const std::vector<int> _sizesOfLayers;
  const std::vector<NodeType> _layersTypes;
  const std::vector<float> _Sparsity;
  const int  _currentBatchSize;


public:
	Network(const std::vector<int> &sizesOfLayers, const std::vector<NodeType> &layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, const std::vector<int> &K, const std::vector<int> &L, const std::vector<int> &RangePow, const std::vector<float> &Sparsity, cnpy::npz_t arr);
	Layer* getLayer(int LayerID);
	int predictClass(Vec2d<int> &inputIndices, Vec2d<float> &inputValues, const Vec2d<int> &labels);
	int ProcessInput(Vec2d<int> &inputIndices, Vec2d<float> &inputValues, const Vec2d<int> &labels, int iter, bool rehash, bool rebuild);
	void saveWeights(std::string file);
	~Network();
};

