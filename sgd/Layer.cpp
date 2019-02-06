#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <climits>
#include "VectorUtils.h"



using namespace std;


Layer::Layer(int noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, Timer *time)
{
	_layerID = layerID;
	_noOfNodes = noOfNodes;
	_Nodes = new Node*[noOfNodes];
	_type = type;
	_time = time;

	if (type == NodeType::Softmax) {
		_noOfActive = floor(_noOfNodes * SMSPARSITY);
		_hashTables = new LSH(SMHASHFUNCTION, SMHASHTABLE);
		_wtaHasher  = new WtaHash(SMHASHFUNCTION * SMHASHTABLE, 20);
	}
	else {
		_noOfActive = floor(_noOfNodes * SPARSITY);
		_hashTables = new LSH(HASHFUNCTION, HASHTABLE);
		_wtaHasher  = new WtaHash(HASHFUNCTION * HASHTABLE, 20);
	}
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, _noOfNodes - 1);
		_randNode = new int[_noOfActive];
		for (int n = 0; n < _noOfActive; n++) {
			_randNode[n] = dis(gen);
			if (_randNode[n] < 0) {
				cout << "wrong generation" << endl;
			}
			if (_randNode[n] > (_noOfNodes - 1)) {
				cout << "wrong generation" << endl;
			}
		}


	//TODO: Initialize Hash Tables and add the nodes. Done by Beidi


	_binids = new int[previousLayerNumOfNodes];
	_wtaHasher->getMap(previousLayerNumOfNodes, _binids);

	for (size_t i = 0; i < noOfNodes; i++)
	{
		_Nodes[i] = new Node(previousLayerNumOfNodes, i, _layerID, MAXSTALEINPUT, _hashTables, _wtaHasher,_binids,  time, type);
		addtoHashTable(_Nodes[i]->_weights, previousLayerNumOfNodes, _Nodes[i]->_bias, i);

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

void Layer::addtoHashTable(float* weights, int length, float bias, int ID)
{
	//LSH logic
	int * hashes = _wtaHasher->getHashEasy(_binids ,weights, length);
	int * hashIndices = _hashTables->hashesToIndex(hashes);
	int * bucketIndices = _hashTables->add(hashIndices, ID+1);

	_Nodes[ID]->_indicesInTables = hashIndices;
	_Nodes[ID]->_indicesInBuckets = bucketIndices;

	delete [] hashes;

}

float Layer::getNomalizationConstant(int inputID)
{
	assert(("Error Call to Normalization Constant for non - softmax layer", _type == NodeType::Softmax));
		return _normalizationConstants[inputID];
}


float innerproduct(int* index1, float* value1, int len1, float* value2){
    float total = 0;
    for (int i=0; i<len1; i++){
        total+=value1[i]*value2[index1[i]];
    }
    return total;
}

void Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int label)
{
	//LSH QueryLogic
	//TODO: it should return the active indices in indices and their activations in outvalues and outindices and outlength.
	//Now compute activations

	//Beidi. Query out all the candidate nodes
	auto t1 = _time->start();
	int * hashes = _wtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], _binids, lengths[layerIndex]);
	auto t2 = _time->start();
	_time->addWta(t1, t2);

	t1 = _time->start();
	int * hashIndices = _hashTables->hashesToIndex(hashes);
	int ** actives = _hashTables->retrieveRaw(hashIndices);
	t2 = _time->start();
	_time->addLsh(t1, t2);

	//Check active & unique & rank
	//I used count + sort for now
	int len;
	int tables = HASHTABLE;
	if (_type == NodeType::Softmax) {
		tables = SMHASHTABLE;
	}
//		len = _noOfNodes;
//		lengths[layerIndex + 1] = _noOfNodes;
//
//		activenodesperlayer[layerIndex + 1] = new int[_noOfNodes]; //assuming not intitialized;
//		for (int i = 0; i < _noOfNodes; i++)
//		{
//			activenodesperlayer[layerIndex + 1][i] = i;
//		}
//	}
//	else {

		t1 = _time->start();
		std::map<int, size_t> counts;
		for (int i = 0; i < tables; i++) {
			if (actives[i] == NULL) {
				continue;
			} else {

				for (int j = 0; j < BUCKETSIZE; j++) {
					int tempID = actives[i][j]-1;
					if (tempID >= 0) {
						Node *tmpNode = _Nodes[tempID];
						int indexInTable = tmpNode->_indicesInTables[i];
						int indexInBucket = tmpNode->_indicesInBuckets[i];
						int chosenBucket = hashIndices[i];

						if ((indexInTable == chosenBucket) & (indexInBucket == j)) {
							counts[tempID] += 1;
						}
					}
				}
			}
		}

//			        cout <<"-----------------layer "<<layerIndex<<"---------------------"<<endl;
//		cout << " Hashtable distribution"<<endl;
//		// _hashTables->count();
//		cout<< " Number selected from hashtable "<< counts.size()<<endl;


//				std::vector<std::pair<int, int>> test;
//		test.reserve(counts.size());
//
//		for (auto &&x : counts)
//			test.emplace_back(-x.second, x.first);
//
//		std::sort(begin(test), end(test));


	if (_type == NodeType::Softmax) {
		if (label > -1) {
			counts[label] = SMHASHTABLE;
		}
	}
		

		for (int i = 0; i < _noOfActive; i++) {
			if (counts.size() >= _noOfActive) {
				break;
			}

			if (counts.count(_randNode[i]) == 0) {
				counts[_randNode[i]] = 0;
			}
		}

		//sorting
		std::vector<std::pair<int, int>> sortNodes;
		sortNodes.reserve(counts.size());

		for (auto &&x : counts)
			sortNodes.emplace_back(-x.second, x.first);

		std::sort(begin(sortNodes), end(sortNodes));

		// pass the active nodes
    	//handle when queried nodes are less than topk
    	len= std::min((int)sortNodes.size(), _noOfActive);
		lengths[layerIndex + 1] = len;
		activenodesperlayer[layerIndex + 1] = new int[len];
		for (int i = 0; i < len; i++) {
			activenodesperlayer[layerIndex + 1][i] = sortNodes[i].second;
		}
		t2 = _time->start();
		_time->addSort(t1, t2);




		//Debugging
//	float mean_compute = 0;
//	for (int s=0; s<counts.size() ;s++){
//		int node_id = sortNodes[s].second;
//		float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], _Nodes[node_id]->_weights);
//		mean_compute+=tmp;
//	}
//
////        cout<<"computed topk = " << mean_compute/len<<endl;
//
//	float total = 0;
//	float overall = 0;
//	vector<pair<float,int> >sortW;
//
//
//	for (int s=0; s<_noOfNodes ;s++){
//		float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], _Nodes[s]->_weights);
////            sortW.push_back(tmp);
//		sortW.push_back (make_pair (-tmp, s));
//		overall+= tmp;
//	}
//	std::sort(begin(sortW), end(sortW));
//
//	int match = 0;
//	for (int s=0; s<_noOfActive; s++){
//		total += sortW[s].first;
//		for (int b=0; b<counts.size(); b++) {
//			if (sortW[s].second ==  sortNodes[b].second) {
//				match++;
//			}
//		}
//	}
//	cout <<" Matched topk" << match*1.0/_noOfActive<<endl;
//
//	cout<<"expected topk mean = " << total/len<<endl;
//	cout <<" overall mean = "<< overall/_noOfNodes <<endl;
//	cout << "ratio of Real-wta/real"<< (total-mean_compute)/total/len<<endl;
//	cout << "--------------------------------------"<<endl;
//
//
//
//
//	cout << "Nodes from real hashtable " << 1.0*counts.size()/_noOfActive << endl;
//





//	}

	activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
	float maxValue = 0;
	if (_type == NodeType::Softmax)
		_normalizationConstants[inputID] = 0;

	for (int i = 0; i < len; i++)
	{
		activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]]->getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
		if(_type == NodeType::Softmax){
			if (activeValuesperlayer[layerIndex + 1][i]>maxValue){
				maxValue = activeValuesperlayer[layerIndex + 1][i];
			}
			if (maxValue<0){

			}
		}

	}


	if(_type == NodeType::Softmax) {
		for (int i = 0; i < len; i++) {
			float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
			activeValuesperlayer[layerIndex + 1][i] =realActivation;
			_Nodes[activenodesperlayer[layerIndex + 1][i]]->_lastActivations[inputID] = realActivation;
			_normalizationConstants[inputID] += realActivation;

		}
		if (isinf(_normalizationConstants[inputID])){

		}
	}



    delete [] hashes;
	delete [] hashIndices;
	delete [] actives;

}

Layer::~Layer()
{
	delete[] _randNode;
	delete[] _hashTables;
	delete[] _wtaHasher;
	delete[] _binids;

	for (size_t i = 0; i < _noOfNodes; i++)
	{
		free(_Nodes[i]);
		if (_type == NodeType::Softmax)
		{
			delete[] _normalizationConstants;
			delete[] _inputIDs;
		}
	}
	delete[] _Nodes;

}
