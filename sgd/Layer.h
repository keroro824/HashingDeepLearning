#pragma once
#include "Node.h"
#include "Config.h"
#include "WtaHash.h"
#include "LSH.h"
//#include "Timer.h"


using namespace std;

class Layer
{
private:
	NodeType _type;
	Node** _Nodes;
	int _layerID, _noOfNodes;
	float* _normalizationConstants;
    int*_inputIDs; //needed for SOFTMAX
    LSH *_hashTables;
    WtaHash *_wtaHasher;
    int * _binids;
	int _noOfActive;
	int * _randNode;
	Timer *_time;

public:
	Layer(int _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, Timer *time);
	Node* getNodebyID(int nodeID);
	Node** getAllNodes();
	void addtoHashTable(float* weights, int length, float bias, int id);
	float getNomalizationConstant(int inputID);
	void queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID, int label);
	~Layer();
};

