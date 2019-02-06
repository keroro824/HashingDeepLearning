#include "Node.h"
#include <random>
#include <math.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

using namespace std;

Node::Node(int dim, int nodeID, int layerID, int maxStaleInputs, LSH *hashTables, WtaHash *wtaHasher, int *binids, Timer *time, NodeType type=NodeType::ReLU)
{
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
	_hashTables = hashTables;
	_wtaHasher = wtaHasher;
	_time = time;
	_binids = binids;


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

	_lastActivations = new float[BATCHSIZE]();
	_lastDeltaforBPs = new float[BATCHSIZE]();
	_lastGradients = new float[BATCHSIZE]();
	_ActiveinputIds = new int[BATCHSIZE]();

	for (size_t i = 0; i < _dim; i++)
	{
		_weights[i] = distribution(generator);
	}
	_bias = 0;
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
	assert(("Input ID more than Batch Size", inputID <= BATCHSIZE));
	
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
	if(isinf(_lastActivations[inputID])){ cout<<_bias<<endl;
		cout <<_lastActivations[inputID]<<endl;
		for (int w=0; w<_dim; w++) {
			cout << "weight" << _weights[w] << endl;
			cout << "vale=ye "<< values[w]<<endl;

		}
	}
	if(isnan(_lastActivations[inputID])){
		cout<<_bias<<endl;
		cout <<_lastActivations[inputID]<<endl;
		for (int w=0; w<_dim; w++) {
			cout << "weight" << _weights[w] << endl;
			cout << "vale=ye "<< values[w]<<endl;

		}
	}
	switch (_type)
	{
	case NodeType::ReLU:
		if (_lastActivations[inputID] < 0)
			_lastActivations[inputID] = 0;
		_lastGradients[inputID] = 1;
		_lastDeltaforBPs[inputID] = 0;
		break;
	case NodeType::Softmax:
//		if (_lastActivations[inputID] > 10) { //clipping //TODO: Check
//			_lastActivations[inputID] = 10;
//		}
//		_lastActivations[inputID] = _lastActivations[inputID];
//		_lastGradients[inputID] = 1;

		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}

	if(isnan(_lastActivations[inputID])){
		cout<<_bias<<endl;
		for (int w=0; w<_dim; w++) {
			cout << "weight" << _weights[w] << endl;
			cout << "vale=ye "<< values[w]<<endl;
		}
	}
	return _lastActivations[inputID];
}

void Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int label)
{ 
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] ==1));
	if (isinf(_lastActivations[inputID])){


	}

	_lastActivations[inputID] /= normalizationConstant + 0.000001;
	//cout << "NodeID LayerID prob " << _IDinLayer << " " << _layerNum << " " << _lastActivations[inputID] << endl;
	//TODO:check  gradient 
	_lastGradients[inputID] = 1;
	if (_IDinLayer == label)
		_lastDeltaforBPs[inputID] = 1 - _lastActivations[inputID];
	else
		_lastDeltaforBPs[inputID] = -_lastActivations[inputID];

	if (isnan(_lastDeltaforBPs[inputID])){


	}
}

void Node::backPropagate(Node** previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
    auto t1 = _time->start();
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
		previousNodes[previousLayerActiveNodeIds[i]]->incrementDelta(inputID, _lastDeltaforBPs[inputID] * _weights[previousLayerActiveNodeIds[i]]);

		if (isnan(previousNodes[previousLayerActiveNodeIds[i]]->getLastActivation(inputID))){


		}

		if (isnan(_lastDeltaforBPs[inputID])){
			cout <<_weights[previousLayerActiveNodeIds[i]]<<endl;
		}
		float grad_t = _lastDeltaforBPs[inputID] * previousNodes[previousLayerActiveNodeIds[i]]->getLastActivation(inputID);
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]]++;

			_adamAvgMom[previousLayerActiveNodeIds[i]] = BETA1 * _adamAvgMom[previousLayerActiveNodeIds[i]] + (1 - BETA1)*grad_t;
			_adamAvgVel[previousLayerActiveNodeIds[i]] = BETA2 * _adamAvgVel[previousLayerActiveNodeIds[i]] + (1 - BETA2)*grad_tsq;

			_adamAvgMom[previousLayerActiveNodeIds[i]] = _adamAvgMom[previousLayerActiveNodeIds[i]] / (1 - pow(BETA1,_t[previousLayerActiveNodeIds[i]]));
			_adamAvgVel[previousLayerActiveNodeIds[i]] = _adamAvgVel[previousLayerActiveNodeIds[i]] / (1 - pow(BETA2,_t[previousLayerActiveNodeIds[i]]));


			_weights[previousLayerActiveNodeIds[i]] += (0.001 / (sqrt(_adamAvgVel[previousLayerActiveNodeIds[i]]) + EPS)) * grad_t;

		}
		else
		{
//			if (abs(grad_t)>abs(_weights[previousLayerActiveNodeIds[i]])*CLIP){
////				grad_t  *=CLIP;
//				grad_t = _weights[previousLayerActiveNodeIds[i]]*CLIP;
//			}

			_weights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;

			if (isnan(_weights[previousLayerActiveNodeIds[i]])){

			}
		}
	}

	if (ADAM)
	{
		_tbias++;
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_adamAvgMombias = BETA1 * _adamAvgMombias + (1 - BETA1)*biasgrad_t;
		_adamAvgVelbias = BETA2 * _adamAvgVelbias + (1 - BETA2)*biasgrad_tsq;

		_adamAvgMombias = _adamAvgMombias / (1 - pow(BETA1, _tbias));
		_adamAvgVelbias = _adamAvgVelbias / (1 - pow(BETA2, _tbias));


		_bias += (0.001 / (sqrt(_adamAvgVelbias) + EPS)) * biasgrad_t;

	}
	else
	{
//		if (abs(_lastDeltaforBPs[inputID]) > abs(_bias) * CLIP) {
////			_bias += learningRate * _lastDeltaforBPs[inputID]*CLIP;
//			_bias += learningRate*_bias*CLIP;
//		}else{
			_bias += learningRate * _lastDeltaforBPs[inputID];
//		}
	}

    auto t2 = _time->start();
    _time->addBack(t1, t2);

	//TODO: UPADTE HashTable
	//TODO: check if index is still valid, if not, update hashtable and index
	t1 = _time->start();
	int * hashes = _wtaHasher->getHashEasy( _binids ,_weights, _dim);
	t2 = _time->start();
	_time->addWta(t1, t2);


	t1 = _time->start();
	int * hashIndices = _hashTables->hashesToIndex(hashes);

	for (int i=0; i<HASHTABLE; i++){

		int curIndex = _indicesInTables[i];
		int futIndex = hashIndices[i];
		int curBucket = _indicesInBuckets[i];

		if (curBucket==-1){
			int bucketIndices = _hashTables->add(i, futIndex, _IDinLayer + 1);
			_indicesInTables[i] = futIndex;
			_indicesInBuckets[i] = bucketIndices;
		}else {
			if ((_hashTables->retrieve(i, curIndex, curBucket) != (_IDinLayer + 1)) | (curIndex != futIndex)) {
				int bucketIndices = _hashTables->add(i, futIndex, _IDinLayer + 1);
				_indicesInTables[i] = futIndex;
				_indicesInBuckets[i] = bucketIndices;
			}
		}
	}

	t2 = _time->start();
	_time->addLsh(t1, t2);

	delete [] hashes;
	delete [] hashIndices;

	_ActiveinputIds[inputID] = 0;
	_lastDeltaforBPs[inputID] = 0;
}

void Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _ActiveinputIds[inputID] == 1));
	auto t1 = _time->start();
	for (int i = 0; i < nnzSize; i++)
	{
		float grad_t = _lastDeltaforBPs[inputID] * nnzvalues[i];
		float grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[nnzindices[i]]++;
			_adamAvgMom[nnzindices[i]] = BETA1 * _adamAvgMom[nnzindices[i]] + (1 - BETA1)*grad_t;
			_adamAvgVel[nnzindices[i]] = BETA2 * _adamAvgVel[nnzindices[i]] + (1 - BETA2)*grad_tsq;

			_adamAvgMom[nnzindices[i]] = _adamAvgMom[nnzindices[i]] / (1 - pow(BETA1, _t[nnzindices[i]]));
			_adamAvgVel[nnzindices[i]] = _adamAvgVel[nnzindices[i]] / (1 - pow(BETA2, _t[nnzindices[i]]));


			_weights[nnzindices[i]] += (0.001 / (sqrt(_adamAvgVel[nnzindices[i]]) + EPS)) * grad_t;

		}
		else
		{
//			if (abs(grad_t)>abs(_weights[nnzindices[i]])*CLIP){
////				grad_t  *=CLIP;
//				grad_t = _weights[nnzindices[i]] * CLIP;
//			}
			_weights[nnzindices[i]] += learningRate * grad_t;


		}
	}

	if (ADAM)
	{
		_tbias++;
		float biasgrad_t = _lastDeltaforBPs[inputID];
		float biasgrad_tsq = biasgrad_t * biasgrad_t;
		_adamAvgMombias = BETA1 * _adamAvgMombias + (1 - BETA1)*biasgrad_t;
		_adamAvgVelbias = BETA2 * _adamAvgVelbias + (1 - BETA2)*biasgrad_tsq;

		_adamAvgMombias = _adamAvgMombias / (1 - pow(BETA1, _tbias));
		_adamAvgVelbias = _adamAvgVelbias / (1 - pow(BETA2, _tbias));


		_bias += (0.001 / (sqrt(_adamAvgVelbias) + EPS)) * biasgrad_t;

	}


	else
	{
//		if (abs(_lastDeltaforBPs[inputID]) > abs(_bias) * CLIP) {
////			_bias += learningRate * _lastDeltaforBPs[inputID]*CLIP;
//			_bias += learningRate*_bias*CLIP;
//		}else{
			_bias += learningRate * _lastDeltaforBPs[inputID];
//		}
	}

	auto t2 = _time->start();
	_time->addBack(t1, t2);
	//TODO: UPDATE HashTable
	//TODO: check if index is still valid, if not, update hashtable and index

	t1 = _time->start();
	int * hashes = _wtaHasher->getHashEasy(_binids,_weights, _dim);
	t2 = _time->start();
	_time->addWta(t1, t2);


	t1 = _time->start();
	int * hashIndices = _hashTables->hashesToIndex(hashes);

	int tables = HASHTABLE;
	if (_type == NodeType::Softmax) {
		tables = SMHASHTABLE;
	}


	for (int i=0; i<tables; i++){

		int curIndex = _indicesInTables[i];
		int futIndex = hashIndices[i];
		int curBucket = _indicesInBuckets[i];

		if (curBucket==-1){
			int bucketIndices = _hashTables->add(i, futIndex, _IDinLayer + 1);
			_indicesInTables[i] = futIndex;
			_indicesInBuckets[i] = bucketIndices;
		}else {
			if ((_hashTables->retrieve(i, curIndex, curBucket) != (_IDinLayer + 1)) | (curIndex != futIndex)) {
				int bucketIndices = _hashTables->add(i, futIndex, _IDinLayer + 1);
				_indicesInTables[i] = futIndex;
				_indicesInBuckets[i] = bucketIndices;
			}
		}
	}

	t2 = _time->start();
	_time->addLsh(t1, t2);

	delete [] hashes;
	delete [] hashIndices;

	_ActiveinputIds[inputID] = 0;//deactivate inputIDs
    _lastDeltaforBPs[inputID] = 0;
}


Node::~Node()
{
	
	delete[] _indicesInTables;
	delete[] _indicesInBuckets;
	delete[] _lastActivations;
	delete[] _lastDeltaforBPs;
	delete[] _lastGradients;
	delete[] _ActiveinputIds;
	delete[] _weights;

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
