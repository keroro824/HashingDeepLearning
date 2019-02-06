#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <climits>



using namespace std;


Layer::Layer(int noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type)
{
	_layerID = layerID;
	_noOfNodes = noOfNodes;
	_Nodes = new Node*[noOfNodes];
	_type = type;

	//TODO: Initialize Hash Tables and add the nodes. Done by Beidi

	for (size_t i = 0; i < noOfNodes; i++)
	{
		_Nodes[i] = new Node(previousLayerNumOfNodes, i, _layerID, MAXSTALEINPUT, type);
	}	

	if (type == NodeType::Softmax)
	{
		_normalizationConstants = new float[BATCHSIZE]();
		_inputIDs = new int[BATCHSIZE]();
	}

}

Node* Layer::getNodebyID(int nodeID)
{
	assert(("nodeID less than _noOfNodes" , nodeID < _noOfNodes));
		return _Nodes[nodeID];
}

Node ** Layer::getAllNodes()
{
	return _Nodes;
}


float Layer::getNomalizationConstant(int inputID)
{
	assert(("Error Call to Normalization Constant for non - softmax layer", _type == NodeType::Softmax));
		return _normalizationConstants[inputID];
}

void Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID) {
    //LSH QueryLogic
    //TODO: it should return the active indices in indices and their activations in outvalues and outindices and outlength.
    //Now compute activations

    //Beidi. Query out all the candidate nodes



    //Right now hardcoding for testing: **************delete ***********
    int len;
    if (layerIndex == 0)
        len = 5;
    else if (layerIndex == 1)
        len = 3;
    else
        len = 10;

    lengths[layerIndex + 1] = len;

    activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
    for (int i = 0; i < len; i++) {
        activenodesperlayer[layerIndex + 1][i] = i;
    }
    //********************** delete till here *************

    activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
    float maxValue = 0;
    if (_type == NodeType::Softmax)
        _normalizationConstants[inputID] = 0;

    for (int i = 0; i < len; i++) {
        activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]]->getActivation(
                activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        if (_type == NodeType::Softmax) {
            if (activeValuesperlayer[layerIndex + 1][i] > maxValue) {
                maxValue = activeValuesperlayer[layerIndex + 1][i];
            }
            if (maxValue < 0) {

            }
        }

    }

    if (_type == NodeType::Softmax) {
        for (int i = 0; i < len; i++) {
            float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
            activeValuesperlayer[layerIndex + 1][i] = realActivation;
            _Nodes[activenodesperlayer[layerIndex + 1][i]]->_lastActivations[inputID] = realActivation;
            _normalizationConstants[inputID] += realActivation;

        }
    }

}

Layer::~Layer()
{

	for (size_t i = 0; i < _noOfNodes; i++)
	{
		free(_Nodes[i]);
		if (_type == NodeType::Softmax)
		{
			delete[] _normalizationConstants;
			delete[] _inputIDs;
		}
	}


}
