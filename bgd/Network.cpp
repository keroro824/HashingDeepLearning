#include "Network.h"
#include <iostream>
#include <math.h>
#include <map>
#define DEBUG 1
using namespace std;

Network::Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchSize, Timer *time)
{
	//TODO: sizesOfLayers[0] should be input dimentions.
	//_BatchSize = batchSize;
	_numberOfLayers = noOfLayers;
	_hiddenlayers = new Layer*[noOfLayers];
	_sizesOfLayers = sizesOfLayers;
	_layersTypes = layersTypes;
	_learningRate = LR;
	_time = time;
	for (int i = 0; i < noOfLayers; i++)
	{
		if (i != 0){
			_hiddenlayers[i] = new Layer(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i], time);
		}
		else{
			_hiddenlayers[i] = new Layer(sizesOfLayers[i], INPUTDIM, i, _layersTypes[i], time);
		}
	}
	cout<<"after finishing layer"<<endl;
	_currentBatchSize = 0;
	_inputIDs = new int[BATCHSIZE]();
}

Layer* Network::getLayer(int LayerID)
{
	if (LayerID < _numberOfLayers)
		return _hiddenlayers[LayerID];
	else
	{
		cout << "LayerID out of bounds" << endl;
		//TODO:Handle
	}
}

int Network::predictClass(int ** inputIndices, float ** inputValues, int * length, int * labels, int batchsize)
{
	int correctPred = 0;
	int correct = 0;
	for (int i = 0; i < batchsize; i++) {
		int **activenodesperlayer = new int *[_numberOfLayers + 1]();
		float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
		int *sizes = new int[_numberOfLayers + 1]();

		activenodesperlayer[0] = inputIndices[i];
		activeValuesperlayer[0] = inputValues[i];
		sizes[0] = length[i];

		//inference
        auto t1 = std::chrono::high_resolution_clock::now();
		for (int j = 0; j < _numberOfLayers; j++) {
			auto t1 = std::chrono::high_resolution_clock::now();
			_hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, -1, true, -1, false);

		}
			auto t2 = std::chrono::high_resolution_clock::now();
			int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//			std::cout << "Hashing Layer "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;



		//compute softmax
//		auto t1 = std::chrono::high_resolution_clock::now();
//		int noOfClasses = _sizesOfLayers[_numberOfLayers - 1];
//		float max_act = ;
//		int predict_class = -1;
//		for (int k = 0; k < noOfClasses; k++) {
//			float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(k)->getLastActivation(i);
//			if (max_act < cur_act) {
//				max_act = cur_act;
//				predict_class = k;
//			}
//		}
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = -22222222;
		int predict_class = -1;
		for (int k = 0; k < noOfClasses; k++) {
//			_hiddenlayers[_numberOfLayers-1]->getNodebyID(activenodesperlayer[_numberOfLayers][k])->ComputeExtaStatsForSoftMax(_hiddenlayers[_numberOfLayers-1]->getNomalizationConstant(i), i, labels[i]);

			float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activenodesperlayer[_numberOfLayers][k])->getLastActivation(i);
			if (max_act < cur_act) {
				max_act = cur_act;
				predict_class = activenodesperlayer[_numberOfLayers][k];
			}

			if (activenodesperlayer[_numberOfLayers][k]==labels[i]){
			    correct++;
			}
		}
//		cout <<max_act<<endl;
		if (predict_class<0) {
			cout << predict_class << endl;
		}


		labels[i];
		if (predict_class==labels[i]){
			correctPred++;
		}
//		auto t2 = std::chrono::high_resolution_clock::now();
//		auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//		std::cout << "Softmax takes" << 1.0 * timeDiffInMiliseconds << std::endl;




		delete[] sizes;
		for (int j = 1; j < _numberOfLayers + 1; j++)
		{
			delete[] activenodesperlayer[j];
			delete[] activeValuesperlayer[j];
		}
		delete[] activenodesperlayer;
		delete[] activeValuesperlayer;
	}

	cout<<"label in hashtable "<< correct <<endl;

	return correctPred;
}

int Network::ProcessInput(int ** inputIndices, float ** inputValues, int * lengths, int * labels, int batchsize, float lrFactor, bool rehash, int iter)
{
	_learningRate*=lrFactor;
	float logloss = 0.0;
	int lowconf = 0;
	float prob = 0.0;
    float tmplr = _learningRate;
	//initialize hashtable
//    vector<pair<int,int> > registerNodes;

	int rightlabel = 0;
//# pragma omp parallel for
	for (int i = 0; i < batchsize; i++) {
        if (ADAM){
            tmplr=_learningRate * sqrt((1-pow(BETA2, iter+1)))/(1-pow(BETA1, iter+1));
        }
		int **activenodesperlayer = new int *[_numberOfLayers + 1]();
		float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
		int *sizes = new int[_numberOfLayers + 1]();

		activenodesperlayer[0] = inputIndices[i];
		activeValuesperlayer[0] = inputValues[i];
		sizes[0] = lengths[i];
		// cout<<"start query"<<endl;

		auto t1 = std::chrono::high_resolution_clock::now();


		for (int j = 0; j < _numberOfLayers; j++) {
				int select = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes,
																	   j, i, labels[i], false, iter, false);
				if (select>0){
					rightlabel++;
				}
		}
        auto t2 = std::chrono::high_resolution_clock::now();
        int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//			std::cout << "Hashing Layer "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;


        //Now backpropagate.


		t1 = std::chrono::high_resolution_clock::now();
		bool noSelect = true;
		for (int j = _numberOfLayers - 1; j >= 0; j--) {
			for (int k = 0; k < sizes[j + 1]; k++) {
				int curNodeId = activenodesperlayer[j + 1][k];
				if (j == _numberOfLayers - 1) {
					//TODO: Compute Extra stats: labels[i];
					_hiddenlayers[j]->getNodebyID(curNodeId)->ComputeExtaStatsForSoftMax(_hiddenlayers[j]->getNomalizationConstant(i), i, labels[i]);

					if (DEBUG && (curNodeId == labels[i]))
					{
						if (_hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i) > 0.5)
						{
							lowconf++;
//							cout<<"Higherst act is "<< _hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i)<<endl;
						}

						prob += _hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i);
						logloss -= log(_hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i) + 0.0000001);
					}

				}

				if (j != 0) {
					_hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagate(
							_hiddenlayers[j - 1]->getAllNodes(), activenodesperlayer[j], sizes[j], tmplr, i);
				} else {
					_hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagateFirstLayer(
							inputIndices[i], inputValues[i], lengths[i], tmplr, i);
				}
			}
		}



		t2 = std::chrono::high_resolution_clock::now();
		timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//		std::cout << "Backprop "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;


		//update hashtable here
		// Flip the _mirrorfloag for all updated nodes
		// Evaluate on validation set, if decrease in accuracy, update hash tables for all those nodes, else flip mirroflag for all of the registerd node.
		// delete registry


		//Free memory to avoid leaks
		delete[] sizes;
		for (int j = 1; j < _numberOfLayers + 1; j++) {
			delete[] activenodesperlayer[j];
			delete[] activeValuesperlayer[j];
		}
		delete[] activenodesperlayer;
		delete[] activeValuesperlayer;




		//TODO: UPADTE HashTable
		//TODO: check if index is still valid, if not, update hashtable and index


	}


	auto t1 = std::chrono::high_resolution_clock::now();
    for (int l = 0; l < _numberOfLayers; l++) {
		if (rehash) {
			_hiddenlayers[l]->_hashTables->clear();
		}
//		# pragma omp parallel for

        for (int m = 0; m < _hiddenlayers[l]->_noOfNodes; m++) {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);

			if (rehash) {

				int *hashes = _hiddenlayers[l]->_wtaHasher->getHashEasy(_hiddenlayers[l]->_binids, tmp->_mirrorWeights,
																		tmp->_dim, _hiddenlayers[l]->_topK);
				int *hashIndices = _hiddenlayers[l]->_hashTables->hashesToIndex(hashes);
				int * bucketIndices = _hiddenlayers[l]->_hashTables->add(hashIndices, m+1);

//				for (int i = 0; i < tables; i++) {
//					int curIndex = tmp->_indicesInTables[i];
//					int futIndex = hashIndices[i];
//					int curBucket = tmp->_indicesInBuckets[i];
//
//					if (curBucket < 0) {
//						int bucketIndices = _hiddenlayers[l]->_hashTables->add(i, futIndex, tmp->_IDinLayer + 1);
//						tmp->_indicesInTables[i] = futIndex;
//						tmp->_indicesInBuckets[i] = bucketIndices;
//					} else {
//						if ((_hiddenlayers[l]->_hashTables->retrieve(i, curIndex, curBucket) != (tmp->_IDinLayer + 1)) |
//							(curIndex != futIndex)) {
//							int bucketIndices = _hiddenlayers[l]->_hashTables->add(i, futIndex, tmp->_IDinLayer + 1);
//							tmp->_indicesInTables[i] = futIndex;
//							tmp->_indicesInBuckets[i] = bucketIndices;
//						}
//					}
//				}

				delete[] hashes;
				delete[] hashIndices;
				delete[] bucketIndices;
			}

			if(ADAM){
                for (int d=0; d< tmp->_dim;d++){
//                    if (tmp->_update[d]>0) {
                    tmp->_adamAvgMom[d] = BETA1 * tmp->_adamAvgMom[d] + (1 - BETA1) * tmp->_t[d];
                    tmp->_adamAvgVel[d] = BETA2 * tmp->_adamAvgVel[d] + (1 - BETA2) * tmp->_t[d] * tmp->_t[d];
                    tmp->_weights[d] += tmplr * tmp->_adamAvgMom[d] / (sqrt(tmp->_adamAvgVel[d]) + EPS);
                    tmp->_t[d] = 0;
//                    }
                }


                tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
                tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
                tmp->_bias += tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
                tmp->_tbias = 0;
			}
            else{
                std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights + (tmp->_dim), tmp->_weights);
                tmp->_bias = tmp->_mirrorbias;
			}

        }
    }
	auto t2 = std::chrono::high_resolution_clock::now();
	int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//	std::cout << "copy weights "<<" takes" << 1.0 * timeDiffInMiliseconds <<rehash << std::endl;

	if (DEBUG)
	{

		cout << "Log Loss after batch processing = " << logloss << endl;
		cout << "Low confidence prediction  = " << lowconf << endl;
		cout << "out of = " << prob << endl;
		cout << "right Label "<<rightlabel*1.0/BATCHSIZE<<endl;
	}
	return logloss;
}


Network::~Network()
{
	delete[] _hiddenlayers;
	delete[] _sizesOfLayers;
	delete[] _layersTypes;
	delete[] _inputIDs;
}
