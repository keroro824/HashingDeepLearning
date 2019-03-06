#pragma once
#include "InputSpecificActiveNetwork.h"
using namespace std;

class InputSpecificActiveNetworkFactory
{
private:
	InputSpecificActiveNetwork* _listInputSpecificActiveNetworks;
	int* _inputIDs;
	int _pallarelBatchSize, _currentSize;

public:
	InputSpecificActiveNetworkFactory(int ID);
	InputSpecificActiveNetwork create(int _layerID, int NodeID, int Dim);
	void updateActiveNodesLayer(int inputID, int LayerID, int* ActiveNodesID);
	int* getLayerActiveNodeIDs(int LayerID);
	~InputSpecificActiveNetworkFactory();
};

