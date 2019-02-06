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

Node::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize)
{
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
    _currentBatchsize = batchsize;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<float> distribution(0.0, 0.01);

	_weights = new float[_dim]();

	if (ADAM)
	{
		_adamAvgMom = new float[_dim]();
		_adamAvgVel = new float[_dim]();
		_t = new int[_dim]();
	}

	_lastActivations = new float[_currentBatchsize]();
	_lastDeltaforBPs = new float[_currentBatchsize]();
	_lastGradients = new float[_currentBatchsize]();
	_ActiveinputIds = new int[_currentBatchsize]();

	for (size_t i = 0; i < _dim; i++)
	{
		_weights[i] = distribution(generator);
//		_weights[i] = 0.05;
	}
//	_bias = distribution(generator);
		_bias = 0.0;
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
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));
	
	//FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too. 
	_ActiveinputIds[inputID] = 1; //activate input

	_lastActivations[inputID] = 0;
	for (int i = 0; i < length; i++)
	{
		_lastActivations[inputID] += _weights[indices[i]] * values[i];

	}
	_lastActivations[inputID] += _bias;


//	if (_lastActivations[inputID] > 10) //clipping //TODO: Check
//		_lastActivations[inputID] = 10;
	//_lastGradients[inputID] = 
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
//        if (_lastActivations[inputID] > 10) //clipping //TODO: Check
//
// 		_lastActivations[inputID] -= 10000;
//		_lastActivations[inputID] = exp(_lastActivations[inputID]);
//
//		_lastGradients[inputID] = 1;
		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}

    if(isinf(_lastActivations[inputID])){

    }

	return _lastActivations[inputID];
}

void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int label)
{ 
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] ==1));


	_lastActivations[inputID] /= normalizationConstant + 0.0000001;
    if(isnan(_lastActivations[inputID])){

    }

	//cout << "NodeID LayerID prob " << _IDinLayer << " " << _layerNum << " " << _lastActivations[inputID] << endl;
	//TODO:check  gradient 
	_lastGradients[inputID] = 1;
	if (_IDinLayer == label)
		_lastDeltaforBPs[inputID] = 1 - _lastActivations[inputID];
	else
		_lastDeltaforBPs[inputID] = -_lastActivations[inputID];
//		_lastDeltaforBPs[inputID] = 0;

	if (inputID==2){

	}

}

void Node::backPropagate(Node** previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
		previousNodes[previousLayerActiveNodeIds[i]]->incrementDelta(inputID, _lastDeltaforBPs[inputID] * _weights[previousLayerActiveNodeIds[i]]);
		float grad_t = _lastDeltaforBPs[inputID] * previousNodes[previousLayerActiveNodeIds[i]]->getLastActivation(inputID);
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]]++;

			_adamAvgMom[previousLayerActiveNodeIds[i]] = BETA1 * _adamAvgMom[previousLayerActiveNodeIds[i]] + (1 - BETA1)*grad_t;
			_adamAvgVel[previousLayerActiveNodeIds[i]] = BETA2 * _adamAvgVel[previousLayerActiveNodeIds[i]] + (1 - BETA2)*grad_tsq;

			_weights[previousLayerActiveNodeIds[i]] +=learningRate* _adamAvgMom[previousLayerActiveNodeIds[i]] / (sqrt(_adamAvgVel[previousLayerActiveNodeIds[i]]) + EPS);

		}
		else
		{
			_weights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		_tbias++;
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_adamAvgMombias = BETA1 * _adamAvgMombias + (1 - BETA1)*biasgrad_t;
		_adamAvgVelbias = BETA2 * _adamAvgVelbias + (1 - BETA2)*biasgrad_tsq;

//		_adamAvgMombias = _adamAvgMombias / (1 - pow(BETA1, _tbias));
//		_adamAvgVelbias = _adamAvgVelbias / (1 - pow(BETA2, _tbias));


		_bias += learningRate*_adamAvgMombias / (sqrt(_adamAvgVelbias) + EPS) ;

	}
	else
		{
			_bias += learningRate * _lastDeltaforBPs[inputID];
		}

	//TODO: UPADTE HashTable
	//TODO: check if index is still valid, if not, update hashtable and index
	_ActiveinputIds[inputID] = 0;
	_lastDeltaforBPs[inputID] = 0;
	_lastActivations[inputID] = 0;

}

void Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));

	for (int i = 0; i < nnzSize; i++)
	{
		float grad_t = _lastDeltaforBPs[inputID] * nnzvalues[i];
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_adamAvgMom[nnzindices[i]] = BETA1 * _adamAvgMom[nnzindices[i]] + (1 - BETA1)*grad_t;
			_adamAvgVel[nnzindices[i]] = BETA2 * _adamAvgVel[nnzindices[i]] + (1 - BETA2)*grad_tsq;
			_weights[nnzindices[i]] += learningRate*_adamAvgMom[nnzindices[i]] / (sqrt(_adamAvgVel[nnzindices[i]]) + EPS);
		}
		else
		{
		    _weights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_adamAvgMombias = BETA1 * _adamAvgMombias + (1 - BETA1)*biasgrad_t;
		_adamAvgVelbias = BETA2 * _adamAvgVelbias + (1 - BETA2)*biasgrad_tsq;

		_bias += learningRate*_adamAvgMombias / (sqrt(_adamAvgVelbias) + EPS);

	}


	else {
	    _bias += learningRate * _lastDeltaforBPs[inputID];
	}

	//TODO: UPDATE HashTable
	//TODO: check if index is still valid, if not, update hashtable and index
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

//	if (ADAM)
//	{
//		delete[] _adamAvgMom;
//		delete[] _adamAvgVel;
//		delete[] _t;
//	}
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
