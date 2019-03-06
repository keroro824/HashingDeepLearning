#include "Node.h"
#include <random>
#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include "Config.h"

using namespace std;

Node::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel)
{
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
    _currentBatchsize = batchsize;

	if (ADAM)
	{
		_adamAvgMom = adamAvgMom;
		_adamAvgVel = adamAvgVel;
		_t = new float[_dim]();

	}

	_lastActivations = new float[_currentBatchsize]();
	_lastDeltaforBPs = new float[_currentBatchsize]();
	_lastGradients = new float[_currentBatchsize]();
	_ActiveinputIds = new int[_currentBatchsize]();

    _weights = weights;
    _bias = bias;
	_mirrorbias = _bias;

}


float Node::getLastActivation(int inputID)
{
	if(_ActiveinputIds[inputID] != 1)
		return 0.0;
	return _lastActivations[inputID];
}


void Node::incrementDelta(int inputID, float incrementValue)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
	if (_lastActivations[inputID]>0)
		_lastDeltaforBPs[inputID] += incrementValue;
}


float Node::getActivation(int* indices, float* values, int length, int inputID)
{
    if (inputID>_currentBatchsize){

    }
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));
	
	//FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too. 
	_ActiveinputIds[inputID] = 1; //activate input

	_lastActivations[inputID] = 0;
	for (int i = 0; i < length; i++)
	{
		_lastActivations[inputID] += _weights[indices[i]] * values[i];

	}
	_lastActivations[inputID] += _bias;

	switch (_type)
	{
	case NodeType::ReLU:
		if (_lastActivations[inputID] < 0) {
            _lastActivations[inputID] = 0;
            _lastGradients[inputID] = 1;
			_lastDeltaforBPs[inputID] = 0;

        }else{
            _lastGradients[inputID] = 0;
		}
		break;
	case NodeType::Softmax:

		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}


	return _lastActivations[inputID];
}


void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
{
	if (_ActiveinputIds[inputID] !=1){

	}
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] ==1));

	_lastActivations[inputID] /= normalizationConstant + 0.0000001;

	//TODO:check  gradient
	_lastGradients[inputID] = 1;
	if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
		_lastDeltaforBPs[inputID] = (1.0/labelsize - _lastActivations[inputID]) / _currentBatchsize;
	}
	else {
		_lastDeltaforBPs[inputID] = (-_lastActivations[inputID]) / _currentBatchsize;
	}
}


void Node::backPropagate(Node** previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
//# pragma omp parallel for
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
		previousNodes[previousLayerActiveNodeIds[i]]->incrementDelta(inputID, _lastDeltaforBPs[inputID] * _weights[previousLayerActiveNodeIds[i]]);
		float grad_t = _lastDeltaforBPs[inputID] * previousNodes[previousLayerActiveNodeIds[i]]->getLastActivation(inputID);
		float grad_tsq = grad_t * grad_t;

		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
		{
			_mirrorbias += learningRate * _lastDeltaforBPs[inputID];
		}

	_ActiveinputIds[inputID] = 0;
	_lastDeltaforBPs[inputID] = 0;
	_lastActivations[inputID] = 0;

}


void Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
//# pragma omp parallel for
	for (int i = 0; i < nnzSize; i++)
	{
		float grad_t = _lastDeltaforBPs[inputID] * nnzvalues[i];
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[nnzindices[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}


	else {
		_mirrorbias += learningRate * _lastDeltaforBPs[inputID];
	}

	_ActiveinputIds[inputID] = 0;//deactivate inputIDs
	_lastDeltaforBPs[inputID] = 0;
	_lastActivations[inputID] = 0;
}


Node::~Node()
{
	
	delete[] _indicesInTables;
	delete[] _indicesInBuckets;
	delete[] _lastActivations;
	delete[] _lastDeltaforBPs;
	delete[] _lastGradients;
	delete[] _ActiveinputIds;

	if (ADAM)
	{
		delete[] _adamAvgMom;
		delete[] _adamAvgVel;
		delete[] _t;
	}
}


// for debugging gradients.
float Node::purturbWeight(int weightid, float delta)
{
	_weights[weightid] += delta;
	return _weights[weightid];
}


float Node::getGradient(int weightid, int inputID, float InputVal)
{
	return -_lastDeltaforBPs[inputID] * InputVal;
}


float* Node::getTestActivation() {
	return _lastActivations;
}


float* Node::getLastDeltaForBPs() {
	return _lastDeltaforBPs;

}
