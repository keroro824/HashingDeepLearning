#include "Network.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#include <omp.h>
#define DEBUG 1
using namespace std;


Network::Network(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, int inputdim,  int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr) {

    _numberOfLayers = noOfLayers;
    _hiddenlayers = new Layer *[noOfLayers];
    _sizesOfLayers = sizesOfLayers;
    _layersTypes = layersTypes;
    _learningRate = lr;
    _currentBatchSize = batchSize;
    _Sparsity = Sparsity;


    for (int i = 0; i < noOfLayers; i++) {
        if (i != 0) {
            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            float* weight, *bias, *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i], _currentBatchSize,  K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        } else {

            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            float* weight, *bias, *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], inputdim, i, _layersTypes[i], _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        }
    }
    cout << "after layer" << endl;
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
            _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], 0,
                    _Sparsity[_numberOfLayers+j], -1);
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
    float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Inference takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

    return correctPred;
}


int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, int iter, bool rehash, bool rebuild) {

    float logloss = 0.0;
    int* avg_retrieval = new int[_numberOfLayers]();

    for (int j = 0; j < _numberOfLayers; j++)
        avg_retrieval[j] = 0;


    if(iter%6946==6945 ){
        //_learningRate *= 0.5;
        _hiddenlayers[1]->updateRandomNodes();
    }
    float tmplr = _learningRate;
    if (ADAM) {
        tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
                (1 - pow(BETA1, iter + 1));
    }else{
//        tmplr *= pow(0.9, iter/10.0);
    }

    int*** activeNodesPerBatch = new int**[_currentBatchSize];
    float*** activeValuesPerBatch = new float**[_currentBatchSize];
    int** sizesPerBatch = new int*[_currentBatchSize];
#pragma omp parallel for
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activeNodesPerBatch[i] = activenodesperlayer;
        activeValuesPerBatch[i] = activeValuesperlayer;
        sizesPerBatch[i] = sizes;

        activenodesperlayer[0] = inputIndices[i];  // inputs parsed from training data file
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];
        int in;
        //auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < _numberOfLayers; j++) {
            in = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], labelsize[i],
                    _Sparsity[j], iter*_currentBatchSize+i);
            avg_retrieval[j] += in;
        }

        //Now backpropagate.
        // layers
        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            Layer* layer = _hiddenlayers[j];
            Layer* prev_layer = _hiddenlayers[j - 1];
            // nodes
            for (int k = 0; k < sizesPerBatch[i][j + 1]; k++) {
                Node* node = layer->getNodebyID(activeNodesPerBatch[i][j + 1][k]);
                if (j == _numberOfLayers - 1) {
                    //TODO: Compute Extra stats: labels[i];
                    node->ComputeExtaStatsForSoftMax(layer->getNomalizationConstant(i), i, labels[i], labelsize[i]);
                }
                if (j != 0) {
                    node->backPropagate(prev_layer->getAllNodes(), activeNodesPerBatch[i][j], sizesPerBatch[i][j], tmplr, i);
                } else {
                    node->backPropagateFirstLayer(inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }
        }
    }
    for (int i = 0; i < _currentBatchSize; i++) {
        //Free memory to avoid leaks
        delete[] sizesPerBatch[i];
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activeNodesPerBatch[i][j];
            delete[] activeValuesPerBatch[i][j];
        }
        delete[] activeNodesPerBatch[i];
        delete[] activeValuesPerBatch[i];
    }

    delete[] activeNodesPerBatch;
    delete[] activeValuesPerBatch;
    delete[] sizesPerBatch;


    auto t1 = std::chrono::high_resolution_clock::now();
    bool tmpRehash;
    bool tmpRebuild;

    for (int l=0; l<_numberOfLayers ;l++) {
        if(rehash & _Sparsity[l]<1){
            tmpRehash=true;
        }else{
            tmpRehash=false;
        }
        if(rebuild & _Sparsity[l]<1){
            tmpRebuild=true;
        }else{
            tmpRebuild=false;
        }
        if (tmpRehash) {
            _hiddenlayers[l]->_hashTables->clear();
        }
        if (tmpRebuild){
            _hiddenlayers[l]->updateTable();
        }
        int ratio = 1;
#pragma omp parallel for
        for (size_t m = 0; m < _hiddenlayers[l]->_noOfNodes; m++)
        {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);
            int dim = tmp->_dim;
            float* local_weights = new float[dim];
            std::copy(tmp->_weights, tmp->_weights + dim, local_weights);

            if(ADAM){
                for (int d=0; d < dim;d++){
                    float _t = tmp->_t[d];
                    float Mom = tmp->_adamAvgMom[d];
                    float Vel = tmp->_adamAvgVel[d];
                    Mom = BETA1 * Mom + (1 - BETA1) * _t;
                    Vel = BETA2 * Vel + (1 - BETA2) * _t * _t;
                    local_weights[d] += ratio * tmplr * Mom / (sqrt(Vel) + EPS);
                    tmp->_adamAvgMom[d] = Mom;
                    tmp->_adamAvgVel[d] = Vel;
                    tmp->_t[d] = 0;
                }

                tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
                tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
                tmp->_bias += ratio*tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
                tmp->_tbias = 0;
                std::copy(local_weights, local_weights + dim, tmp->_weights);
            }
            else
            {
                std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
                tmp->_bias = tmp->_mirrorbias;
            }
            if (tmpRehash) {
                int *hashes;
                if(HashFunction==1) {
                    hashes = _hiddenlayers[l]->_wtaHasher->getHash(local_weights);
                }else if (HashFunction==2){
                    hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(local_weights, dim, TOPK);
                }else if (HashFunction==3){
                    hashes = _hiddenlayers[l]->_MinHasher->getHashEasy(_hiddenlayers[l]->_binids, local_weights, dim, TOPK);
                }else if (HashFunction==4){
                    hashes = _hiddenlayers[l]->_srp->getHash(local_weights, dim);
                }

                int *hashIndices = _hiddenlayers[l]->_hashTables->hashesToIndex(hashes);
                int * bucketIndices = _hiddenlayers[l]->_hashTables->add(hashIndices, m+1);

                delete[] hashes;
                delete[] hashIndices;
                delete[] bucketIndices;
            }


            delete[] local_weights;
        }
    }

    if (DEBUG&rehash) {
        cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
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
}
