#include "Network.h"
#include <iostream>
#include <math.h>

#define DEBUG 1
using namespace std;

Network::Network(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize) {
    //TODO: sizesOfLayers[0] should be input dimentions.
    //_BatchSize = batchSize;
    _numberOfLayers = noOfLayers;
    _hiddenlayers = new Layer *[noOfLayers];
    _sizesOfLayers = sizesOfLayers;
    _layersTypes = layersTypes;
    _learningRate = LR;
    for (int i = 0; i < noOfLayers; i++) {
        if (i != 0) {
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i]);
        } else {
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], INPUTDIM, i, _layersTypes[i]);
        }
    }
    cout << "after layer" << endl;
    _currentBatchSize = 0;
    _inputIDs = new int[BATCHSIZE]();
}

Layer *Network::getLayer(int LayerID) {
    if (LayerID < _numberOfLayers)
        return _hiddenlayers[LayerID];
    else {
        cout << "LayerID out of bounds" << endl;
        //TODO:Handle
    }
}

float *Network::predictProb(int *inputIndices, int *inputValue, int lenght) {
    return nullptr;
}


int Network::predictClass(int **inputIndices, float **inputValues, int *length, int *labels, int batchsize) {
    int correctPred = 0;
    for (int i = 0; i < batchsize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = length[i];

        //inference
        for (int j = 0; j < _numberOfLayers; j++) {
            _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j,
                                                                   i);
        }

        //compute softmax
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = 0;
        int predict_class = -1;
        for (int k = 0; k < noOfClasses; k++) {
            float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activenodesperlayer[_numberOfLayers][k])->getLastActivation(i);
            if (max_act < cur_act) {
                max_act = cur_act;
                predict_class = activenodesperlayer[_numberOfLayers][k];
            }
        }
        if (predict_class == labels[i]) {
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
    return correctPred;
}


int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int *labels, int batchsize,
                          float lrFactor) {
    _learningRate *= lrFactor;

    float logloss = 0.0;
    int lowconf = 0;
    float prob = 0.0;
//    # pragma omp parallel for
    for (int i = 0; i < batchsize; i++) {

        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];
        // cout<<"start query"<<endl;

        auto t1 = std::chrono::high_resolution_clock::now();


        for (int j = 0; j < _numberOfLayers; j++) {
            _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j,
                                                                   i);
        }


        //Now backpropagate.


        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            for (int k = 0; k < sizes[j + 1]; k++) {
                if (j == _numberOfLayers - 1) {
                    //TODO: Compute Extra stats: labels[i];
                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->ComputeExtaStatsForSoftMax(
                            _hiddenlayers[j]->getNomalizationConstant(i), i, labels[i]);

                    //activeValuesperlayer[j + 1][k] = _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->getLastActivation(i);

                    if (DEBUG && (activenodesperlayer[j + 1][k] == labels[i])) {
                        //cout << "Label " << k  << "Label[i]" << labels[i] << " PredictedProb " << _hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i) << endl;
                        if (_hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i) < 0.5)
                            lowconf++;
                        prob += _hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i);
                        logloss -= log(
                                _hiddenlayers[_numberOfLayers - 1]->getNodebyID(labels[i])->getLastActivation(i) +
                                0.0000001);
                    }
                }

                if (j != 0) {
                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagate(
                            _hiddenlayers[j - 1]->getAllNodes(), activenodesperlayer[j], sizes[j], _learningRate, i);
                } else {
                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagateFirstLayer(
                            inputIndices[i], inputValues[i], lengths[i], _learningRate, i);
                }
            }

        }



        //Free memory to avoid leaks
        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activenodesperlayer[j];
            delete[] activeValuesperlayer[j];
        }
        delete[] activenodesperlayer;
        delete[] activeValuesperlayer;



    }

    for (int l=0; l<_numberOfLayers ;l++) {
//        # pragma omp parallel for
        for (int m=0; m< _hiddenlayers[l]->_noOfNodes; m++) {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);
            std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
            tmp->_bias = tmp->_mirrorbias;
        }

    }


    if (DEBUG) {

        cout << "Log Loss after batch processing = " << logloss << endl;
        cout << "Low confidence prediction  = " << lowconf << endl;
        cout << "out of = " << prob << endl;
    }
    return logloss;
}

Network::~Network() {
    delete[] _hiddenlayers;
    delete[] _sizesOfLayers;
    delete[] _layersTypes;
    delete[] _inputIDs;
}
