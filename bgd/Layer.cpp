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
		_noOfActive = floor(_noOfNodes * SMSPARSITYTRAIN);
		_topK = previousLayerNumOfNodes * SMSPARSITYTRAIN;
	}
	else {
        _noOfActive = floor(_noOfNodes * SPARSITY);
		_topK = previousLayerNumOfNodes * SPARSITY;
    }


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, _noOfNodes - 1);
	_randNode = new int[_noOfNodes];
	for (int n = 0; n < _noOfNodes; n++) {
		_randNode[n] = n;
	}

	std::random_shuffle ( _randNode, _randNode+_noOfNodes );


	//TODO: Initialize Hash Tables and add the nodes. Done by Beidi

	if (type == NodeType::Softmax) {
		_hashTables = new LSH(SMHASHFUNCTION, SMHASHTABLE, SMRANGEROW);
		_wtaHasher = new WtaHash(SMHASHFUNCTION * SMHASHTABLE, 20);
	}else{
		_hashTables = new LSH(HASHFUNCTION, HASHTABLE, RANGEROW);
		_wtaHasher = new WtaHash(HASHFUNCTION * HASHTABLE, 20);
	}



	_binids = new int[previousLayerNumOfNodes];
	_wtaHasher->getMap(previousLayerNumOfNodes, _binids, previousLayerNumOfNodes);

	for (size_t i = 0; i < noOfNodes; i++)
	{
		_Nodes[i] = new Node(previousLayerNumOfNodes, i, _layerID, MAXSTALEINPUT, type);
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
	int * hashes = _wtaHasher->getHashEasy(_binids ,weights, length, _topK);
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

float collision(int* hashes, int* table_hashes, int k, int l){
    int cp = 0;
    for (int i=0; i<l; i=i+k){
        int tmp = 0;
        for (int j=i; j< i+k;j++){
            if(hashes[j]==table_hashes[j]){
                tmp++;
            }
        }
        if (tmp==k){
            cp++;
        }
    }
    return cp*1.0/(l/k);
}


int Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int label, bool on, int iter, bool test) {
	//LSH QueryLogic
	//TODO: it should return the active indices in indices and their activations in outvalues and outindices and outlength.
	//Now compute activations


	int len;
	int included = 0;
	//Beidi. Query out all the candidate nodes


//	if(_type == NodeType::Softmax & !on) {
//		bool getCorrect = false;
//		std::random_shuffle ( _randNode, _randNode+_noOfNodes );
//		len = _noOfActive;
//		lengths[layerIndex + 1] = len;
//		activenodesperlayer[layerIndex + 1] = new int[len];
//		for (int i = 0; i < len; i++) {
//			activenodesperlayer[layerIndex + 1][i] = _randNode[i];
//			if (label==_randNode[i]){
//				getCorrect=true;
//			}
//		}
//		if(!getCorrect){
//			activenodesperlayer[layerIndex + 1][len-1] = label;
//		}
//
//	}else{
//
//		len = _noOfNodes;
//		lengths[layerIndex + 1] = len;
//		activenodesperlayer[layerIndex + 1] = new int[len];
//		for (int i = 0; i < len; i++) {
//			activenodesperlayer[layerIndex + 1][i] = i;
//		}
//
//	}





	if(_type == NodeType::Softmax & on){

		len = _noOfNodes;
		lengths[layerIndex + 1] = len;
		activenodesperlayer[layerIndex + 1] = new int[len];
		for (int i = 0; i < len; i++) {
			activenodesperlayer[layerIndex + 1][i] = i;
		}

	}
	else {
		auto t1 = _time->start();
		int *hashes = _wtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], _binids,
										  lengths[layerIndex]);

		auto t2 = _time->start();
		_time->addWta(t1, t2);

		t1 = _time->start();
		int *hashIndices = _hashTables->hashesToIndex(hashes);
		int **actives = _hashTables->retrieveRaw(hashIndices);
		t2 = _time->start();
		_time->addLsh(t1, t2);

		//Check active & unique & rank
		//I used count + sort for now
		int tables = HASHTABLE;
		if (_type == NodeType::Softmax){
			tables = SMHASHTABLE;
		}

//		t1 = std::chrono::high_resolution_clock::now();
		std::map<int, size_t> counts;
		for (int i = 0; i < tables; i++) {
			if (actives[i] == NULL) {
				continue;
			} else {
				for (int j = 0; j < BUCKETSIZE; j++) {
					int tempID = actives[i][j] - 1;
					if (tempID >= 0) {
//						Node *tmpNode = _Nodes[tempID];
//						int indexInTable = tmpNode->_indicesInTables[i];
//						int indexInBucket = tmpNode->_indicesInBuckets[i];
//						int chosenBucket = hashIndices[i];
//
//						if ((indexInTable == chosenBucket) & (indexInBucket == j)) {
							counts[tempID] += 1;
//						}
					}
					else{
						break;
					}
				}
			}
		}


//	if (_type == NodeType::Softmax) {
//		cout << "-----------------layer " << layerIndex << "---------------------" << endl;
////		cout << " Hashtable distribution" << endl;
////		_hashTables->count();
//		cout << " Number selected from hashtable " << counts.size() << endl;
//	}

        int tmpnoOfActive = _noOfActive;
		if (_type == NodeType::Softmax) {
			if (label >= 0) {
					counts[label] = SMHASHTABLE;
			}
			if (test){
			    tmpnoOfActive = floor(_noOfNodes*SMSPARSITYTEST);
			}
		}

		srand (time(NULL));
		int start = rand() % tmpnoOfActive;

		for (int i = start; i < tmpnoOfActive; i++) {
			if (counts.size() >= tmpnoOfActive) {
				break;
			}
			if (counts.count(_randNode[i]) == 0) {
				counts[_randNode[i]] = 0;
			}
		}

		if (counts.size()<tmpnoOfActive){
			for (int i = 0; i < tmpnoOfActive; i++) {
				if (counts.size() >= tmpnoOfActive) {
					break;
				}
				if (counts.count(_randNode[i]) == 0) {
					counts[_randNode[i]] = 0;
				}
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
		len = std::min((int) sortNodes.size(), tmpnoOfActive);
		lengths[layerIndex + 1] = len;
		activenodesperlayer[layerIndex + 1] = new int[len];
		for (int i = 0; i < len; i++) {
			activenodesperlayer[layerIndex + 1][i] = sortNodes[i].second;
			if ((_type == NodeType::Softmax) & activenodesperlayer[layerIndex + 1][i]==label){
			    included=1;
			}
		}
		t2 = _time->start();
		_time->addSort(t1, t2);


		if (_type == NodeType::Softmax) {
			if (iter>=0) {
				if (included != 1) {
					cout << "---------------" << endl;
					cout << iter << endl;
					cout << counts[label] << endl;
					cout << sortNodes[0].first << endl;
					cout << sortNodes[0].second << endl;
				}
			}
		}
		delete[] hashes;
		delete[] hashIndices;
		delete[] actives;


		//Debugging
		if (iter == 5000000000000000000000)
		{
			if (_layerID==2) {

				float mean_compute = 0;
				int *hashes = _wtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
												  _binids,
												  lengths[layerIndex]);


//				cout<<" query hash ";
//				for (int h=0; h<SMHASHFUNCTION * SMHASHTABLE; h++){
//				    cout <<hashes[h]<<" ";
//				}
//				cout <<endl;
//
//
//                cout<<" query ";
//                for (int h=0; h<lengths[layerIndex]; h++){
//                    cout <<activenodesperlayer[layerIndex][h]<< ":"<<  activeValuesperlayer[layerIndex][h]<<" ";
//                }
//                cout <<endl;


				for (int s = 0; s < _noOfNodes; s++) {
					int node_id = s;
					float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
											 lengths[layerIndex], _Nodes[node_id]->_weights);
					mean_compute += tmp;

					int *table_hashes = _wtaHasher->getHashEasy(_binids, _Nodes[node_id]->_weights,
																_Nodes[node_id]->_dim, _topK);





//                    cout<<" node hash ";
//                    for (int h=0; h<SMHASHFUNCTION * SMHASHTABLE; h++){
//                        cout <<table_hashes[h]<<" ";
//                    }
//                    cout <<endl;
//
//
//                    cout<<" node ";
//                    for (int h=0; h<_Nodes[node_id]->_dim; h++){
//                        cout <<_Nodes[node_id]->_weights[h]<<" ";
//                    }
//                    cout <<endl;
					float cp = collision(hashes, table_hashes, 1, SMHASHTABLE*SMHASHFUNCTION);
					cout <<"k=1 "<< tmp << " " << cp << endl;
					cp = collision(hashes, table_hashes, 2, SMHASHTABLE*SMHASHFUNCTION);
					cout <<"k=2 "<< tmp << " " << cp << endl;
					cp = collision(hashes, table_hashes, 3, SMHASHTABLE*SMHASHFUNCTION);
					cout <<"k=3 "<< tmp << " " << cp << endl;
//                    exit(0);

				}
				exit(0);
			}
		}


//		if (_type == NodeType::Softmax) {
//			float mean_compute = 0;
//			for (int s = 0; s < len; s++) {
//				int node_id = activenodesperlayer[layerIndex + 1][s];
//				float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
//										 lengths[layerIndex], _Nodes[node_id]->_weights);
//				mean_compute += tmp;
//			}
//
////        cout<<"computed topk = " << mean_compute/len<<endl;
//
//			float total = 0;
//			float overall = 0;
//			vector<pair<float, int> > sortW;
//
//
//			for (int s = 0; s < _noOfNodes; s++) {
//				float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
//										 lengths[layerIndex], _Nodes[s]->_weights);
////            sortW.push_back(tmp);
//				sortW.push_back(make_pair(tmp, s));
//				overall += tmp;
//			}
//			std::sort(begin(sortW), end(sortW));
//
//			int match = 0;
//			for (int s = 0; s < len; s++) {
//				total += sortW[_noOfNodes - 1 - s].first;
//				for (int b = 0; b < len; b++) {
//					if (sortW[_noOfNodes - 1 - s].second == activenodesperlayer[layerIndex + 1][b]) {
//						match++;
//					}
//				}
//			}
//			cout << " Matched topk" << match * 1.0 / len << endl;
//			cout << "Max inner " << sortW[_noOfNodes - 1].first <<endl;
//			cout << "hashing topk mean = " << mean_compute / len << endl;
//			cout << "expected topk mean = " << total / len << endl;
//			cout << " overall mean = " << overall / _noOfNodes << endl;
//			cout << "--------------------------------------" << endl;
//		}

	}

	activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
	float maxValue = 0;
	if (_type == NodeType::Softmax)
		_normalizationConstants[inputID] = 0;



	auto t1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < len; i++) {

		activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]]->getActivation(
				activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);

		if (_type == NodeType::Softmax) {
			if (activeValuesperlayer[layerIndex + 1][i] > maxValue) {
				maxValue = activeValuesperlayer[layerIndex + 1][i];
			}
		}

	}

//	if (_type == NodeType::Softmax) {
//		auto t2 = std::chrono::high_resolution_clock::now();
//		int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//		std::cout << "Last Layer takes" << 1.0 * timeDiffInMiliseconds << std::endl;
//	}

	if (_type == NodeType::Softmax) {

		for (int i = 0; i < len; i++) {
			float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
			activeValuesperlayer[layerIndex + 1][i] = realActivation;
			_Nodes[activenodesperlayer[layerIndex + 1][i]]->_lastActivations[inputID] = realActivation;
			_normalizationConstants[inputID] += realActivation;


		}
//        _normalizationConstants[inputID] += exp(-maxValue)*(_noOfNodes-len);
	}



	return included;

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
