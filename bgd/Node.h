#pragma once
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include "Config.h"


using namespace std;

enum NodeType
{ ReLU, Softmax};

class Node
{
private:

//	int _dim, _layerNum, _IDinLayer;

	NodeType _type;

public:
	float* _lastDeltaforBPs;
	int* _ActiveinputIds;
	float* _lastActivations;
	int _dim, _layerNum, _IDinLayer;
	int* _indicesInTables;
	int* _indicesInBuckets;
	float* _weights;
	float* _mirrorWeights;
	float _bias =0;
	float _mirrorbias =0;
	float _tbias = 0;
	float* _t;
	float* _adamAvgMom;
	float* _adamAvgVel;
	float _adamAvgMombias=0;
	float _adamAvgVelbias=0;
	float* getTestActivation();
	float* getLastDeltaForBPs();

	Node(int dim, int nodeID, int layerId, int maxStaleInputs, NodeType type);
	float getLastActivation(int inputID);
	void incrementDelta(int inputID, float incrementValue);
	float getActivation(int* indices, float* values, int length, int inputID);
	void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int label);
	void backPropagate(Node** previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
	void backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID);
	~Node();
	//Node clone();

	//only for debugging
	float purturbWeight(int weightid, float delta);
	float getGradient(int weightid, int inputID, float InputVal);
};