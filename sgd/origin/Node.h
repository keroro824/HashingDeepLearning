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
	


	float* _lastDeltaforBPs;
	float* _lastGradients;
	int* _ActiveinputIds;
	NodeType _type;

public:
    float* _lastActivations;
    int _dim, _layerNum, _IDinLayer;
	int* _indicesInTables;
	int* _indicesInBuckets;
	float* _weights;
	float* _adamAvgMom;
	float* _adamAvgVel;
	int* _t; //for adam
	float _bias =0;
	int _tbias;
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