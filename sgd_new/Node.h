#pragma once
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH.h"
#include "DensifiedWtaHash.h"

using namespace std;

enum NodeType
{ ReLU, Softmax};

class Node
{
private:
	



	float* _lastGradients;
	int* _ActiveinputIds;
	NodeType _type;

public:
	float* _lastDeltaforBPs;
    float* _lastActivations;
    int _currentBatchsize;
    int _dim, _layerNum, _IDinLayer;
	int* _indicesInTables;
	int* _indicesInBuckets;
	float* _weights;
	float* _mirrorWeights;
	float* _adamAvgMom;
	float* _adamAvgVel;
	int* _update;
	float _bias =0;
	float _tbias = 0;
	float _adamAvgMombias=0;
	float _adamAvgVelbias=0;
	float _mirrorbias =0;
	float* getTestActivation();
	float* getLastDeltaForBPs();

	LSH *_hashTables;
	WtaHash *_wtaHasher;
	DensifiedMinhash *_MinHasher;
	SparseRandomProjection *_srp;
	DensifiedWtaHash *_dwtaHasher;
	int * _binids;
	int _L;
	Node(int dim, int nodeID, int layerId, NodeType type, int batchsize, float *weights, float bias);
    void updateWeights(float* newWeights, float newbias);
	void init(LSH *_hashTables, WtaHash *_wtaHasher, DensifiedMinhash *_MinHasher, SparseRandomProjection *_srp, DensifiedWtaHash *_dwtaHasher, int * _binids, int L );
	float getLastActivation(int inputID);
	void incrementDelta(int inputID, float incrementValue);
	float getActivation(int* indices, float* values, int length, int inputID);
	void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize);
	void backPropagate(Node** previousNodes,int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID, bool rehash);
	void backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID, bool rehash);
	~Node();
	//Node clone();

	//only for debugging
	float purturbWeight(int weightid, float delta);
	float getGradient(int weightid, int inputID, float InputVal);
};