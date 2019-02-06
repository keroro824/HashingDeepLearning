#pragma once
#include "Node.h"

using namespace std;

class Layer
{
private:
	NodeType _type;
	Node** _Nodes;

	float* _normalizationConstants;
    int*_inputIDs; //needed for SOFTMAX

public:
	int _layerID, _noOfNodes;
	Layer(int _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize);
	Node* getNodebyID(int nodeID);
	Node** getAllNodes();
	float getNomalizationConstant(int inputID);
	void queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID);
	~Layer();
};

