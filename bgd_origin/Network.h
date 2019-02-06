#pragma once
#include "Layer.h"
#include <chrono>
using namespace std;

class Network
{
private:
	Layer** _hiddenlayers;
	float _learningRate;
	int _numberOfLayers;
	int* _sizesOfLayers;
	NodeType* _layersTypes;

	int* _inputIDs;
	int _BatchSize, _currentBatchSize;


public:
	Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim);
	Layer* getLayer(int LayerID);
	float* predictProb(int * inputIndices, int * inputValue, int length);
	 int predictClass(int ** inputIndices, float ** inputValues, int * length, int * labels);
	int ProcessInput(int** inputIndices, float** inputValues, int* lengths, int* label, int iter);
	~Network();
};

