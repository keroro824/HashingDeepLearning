#include "Network.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#define DEBUG 1
using namespace std;

Network::Network(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, int inputdim,  int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr) {
    //TODO: sizesOfLayers[0] should be input dimentions.
    //_BatchSize = batchSize;
    _numberOfLayers = noOfLayers;
    _hiddenlayers = new Layer *[noOfLayers];
    _sizesOfLayers = sizesOfLayers;
    _layersTypes = layersTypes;
    _learningRate = lr;
    _currentBatchSize = batchSize;
    _Sparsity = Sparsity;


    for (int i = 0; i < noOfLayers; i++) {
        if (i != 0) {
            cnpy::NpyArray weightArr, biasArr;
            float* weight, *bias, *weightT ;
            if(LOADWEIGHT){
//                weightArr = arr["weight_"+to_string(i)];
//                weightT = weightArr.data<float>();
//                int a, b, p, k, r, c;
//                r = weightArr.shape[0];
//                c = weightArr.shape[1];
//                weight = new float[r*c];
//                for(k=0; k< r*c ;++k)
//                {
//                    a = k/c;
//                    b = k - a*c;
//                    p = b*r + a;
//                    weight[p] = weightT[k];
//                }
//
//                biasArr = arr["bias_"+to_string(i)];
//                bias = biasArr.data<float>();
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();

            }
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i], _currentBatchSize,  K[i], L[i], RangePow[i], Sparsity[i], weight, bias);
        } else {

            cnpy::NpyArray weightArr, biasArr;
            float* weight, *bias, *weightT ;
            if(LOADWEIGHT){
//                weightArr = arr["weight_"+to_string(i)];
//                weightT = weightArr.data<float>();
//                int a, b, p, k, r, c;
//                r = weightArr.shape[0];
//                c = weightArr.shape[1];
//                weight = new float[r*c];
//                for(k=0; k< r*c ;++k)
//                {
//                    a = k/c;
//                    b = k - a*c;
//                    p = b*r + a;
//                    weight[p] = weightT[k];
//                }
//
//                biasArr = arr["bias_"+to_string(i)];
//                bias = biasArr.data<float>();
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();
            }
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], inputdim, i, _layersTypes[i], _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i], weight, bias);
        }
    }
    cout << "after layer" << endl;
    _inputIDs = new int[_currentBatchSize]();
}

Layer *Network::getLayer(int LayerID) {
    if (LayerID < _numberOfLayers)
        return _hiddenlayers[LayerID];
    else {
        cout << "LayerID out of bounds" << endl;
        //TODO:Handle
    }
}



int Network::predictClass(int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize) {
    int correctPred = 0;
    int inhashtable = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = length[i];

        //inference
        for (int j = 0; j < _numberOfLayers; j++) {
            int tmp = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], 0, _Sparsity[_numberOfLayers+j], -1);
            inhashtable+=tmp;
        }

        //compute softmax
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = -222222222;
        int predict_class = -1;
        for (int k = 0; k < noOfClasses; k++) {
            float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activenodesperlayer[_numberOfLayers][k])->getLastActivation(i);
            if (max_act < cur_act) {
                max_act = cur_act;
                predict_class = activenodesperlayer[_numberOfLayers][k];
            }
        }

        if (std::find (labels[i], labels[i]+labelsize[i], predict_class)!= labels[i]+labelsize[i]) {
            correctPred++;
        }


        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activenodesperlayer[j];
            delete[] activeValuesperlayer[j];
        }
        delete[] activenodesperlayer;
        delete[] activeValuesperlayer;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Inference " <<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

//    cout<<inhashtable<<endl;
    return correctPred;
}

int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, int iter, bool rehash) {


    float logloss = 0.0;
    int lowconf = 0;
    float prob = 0.0;
    int* avg_retrieval = new int[_numberOfLayers]();

    int check = 0;
//    if(iter%400==399 && _learningRate>0.000009){
//        _learningRate *= 0.2;
//    }
    float tmplr = _learningRate;

    if (ADAM) {
        tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
                (1 - pow(BETA1, iter + 1));
    }else{
//        tmplr *= pow(0.9, iter/10.0);
    }
    # pragma omp parallel for
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];
        // cout<<"start query"<<endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < _numberOfLayers; j++) {


            int in = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], labelsize[i], _Sparsity[j], iter*_currentBatchSize+i);
//            if (labelsize[i]==in) {
//                check ++;
//            }
            avg_retrieval[j]+=in;


        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//        std::cout << "forward "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;


        //Now backpropagate.

        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            for (int k = 0; k < sizes[j + 1]; k++) {
                if (j == _numberOfLayers - 1) {
                    //TODO: Compute Extra stats: labels[i];
                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->ComputeExtaStatsForSoftMax(
                            _hiddenlayers[j]->getNomalizationConstant(i), i, labels[i], labelsize[i]);

//                    if (DEBUG && find(labels[i], labels[i]+labelsize[i], activenodesperlayer[j + 1][k]) != labels[i]+labelsize[i]) {
//                        logloss -= log(
//                                _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activenodesperlayer[j + 1][k])->getLastActivation(i) +
//                                0.0000001);
//                    }
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

//        cout<<endl;

        //Free memory to avoid leaks
        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activenodesperlayer[j];
            delete[] activeValuesperlayer[j];
        }
        delete[] activenodesperlayer;
        delete[] activeValuesperlayer;
    }

//    cout<<" number of " << check <<endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    bool tmpRehash;

    for (int l=0; l<_numberOfLayers ;l++) {
        if(rehash & _Sparsity[l]<1){
            tmpRehash=true;
        }else{
            tmpRehash=false;
        }
        if (rehash) {
            _hiddenlayers[l]->_hashTables->clear();
        }
        int counthash = 0;
        # pragma omp parallel for reduction(+:counthash)
        for (int m=0; m< _hiddenlayers[l]->_noOfNodes; m++) {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);

            if(ADAM){
                for (int d=0; d< tmp->_dim;d++){
                        tmp->_adamAvgMom[d] = BETA1 * tmp->_adamAvgMom[d] + (1 - BETA1) * tmp->_t[d];
                        tmp->_adamAvgVel[d] = BETA2 * tmp->_adamAvgVel[d] + (1 - BETA2) * tmp->_t[d] * tmp->_t[d];
                        tmp->_weights[d] += tmplr * tmp->_adamAvgMom[d] / (sqrt(tmp->_adamAvgVel[d]) + EPS);
                        tmp->_t[d] = 0;
                }


                    tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
                    tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
                    tmp->_bias += tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
                    tmp->_tbias = 0;


            }
            else{
                std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
                tmp->_bias = tmp->_mirrorbias;
            }
            if (tmpRehash) {


                int *hashes;
                if(HashFunction==1) {
                    hashes = _hiddenlayers[l]->_wtaHasher->getHash(tmp->_weights);
                }else if (HashFunction==2){
                    hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(tmp->_weights, tmp->_dim, TOPK);
                }else if (HashFunction==3){
                    hashes = _hiddenlayers[l]->_MinHasher->getHashEasy(_hiddenlayers[l]->_binids, tmp->_weights, tmp->_dim, TOPK);
                }else if (HashFunction==4){
                    hashes = _hiddenlayers[l]->_srp->getHash(tmp->_weights, tmp->_dim);
                }



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
        }


    }

    auto t2 = std::chrono::high_resolution_clock::now();
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//            std::cout << "Layer " <<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

//    if (rehash) {
//        _hiddenlayers[_numberOfLayers - 1]->_hashTables->count();
//    }
    if(iter==100000000000) {
        float sum = 0;
//        for (int l=0; l<lengths[i]; l++){
//            sum+= inputValues[i][l];
//            cout<<inputValues[i][l]<<" ";
//        }
//        cout<<endl;
//        cout <<sum<<endl;
//        cout<<labels[0][0]<<labels[1][0]<<labels[2][0]<<labels[3][0]<<labels[4][0]<<endl;
//        cout << "Iteration " << i << endl;
        for (int j = _numberOfLayers-1; j < _numberOfLayers; j++) {
            cout << "Layer " << j << endl;
            Node *tmp = _hiddenlayers[j]->getNodebyID(0);
            cout << "Weights for Node" << 0 << " ";
            for (int m = 0; m < tmp->_dim; m++) {
                cout << tmp->_weights[m] << " ";
//                    cout<<tmp->_t[m]<<" ";
            }
            cout << endl;
            for (int w = 0; w < 100; w++) {
                Node *tmp = _hiddenlayers[j]->getNodebyID(w);


                cout << "Bias " << tmp->_bias << endl;

            }
        }

        exit(0);
    }



    if (DEBUG&rehash) {

        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize<<" Log Loss = " << logloss/_currentBatchSize << endl;
//        cout << "Low confidence prediction  = " << lowconf << endl;
//        cout << "out of = " << prob << endl;
    }
    return logloss;
}

void Network::saveWeights(string file)
{
    for (int i=0; i< _numberOfLayers; i++){
        _hiddenlayers[i]->saveWeights(file);
    }
}

Network::~Network() {

    delete[] _sizesOfLayers;
    for (int i=0; i< _numberOfLayers; i++){
        delete _hiddenlayers[i];
    }
    delete[] _hiddenlayers;
    delete[] _layersTypes;
    delete[] _inputIDs;
}
