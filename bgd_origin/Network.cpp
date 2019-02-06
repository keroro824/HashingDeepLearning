#include "Network.h"
#include <iostream>
#include <math.h>
#include "Config.h"
#define DEBUG 1
using namespace std;

Network::Network(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, int inputdim) {
    //TODO: sizesOfLayers[0] should be input dimentions.
    //_BatchSize = batchSize;
    _numberOfLayers = noOfLayers;
    _hiddenlayers = new Layer *[noOfLayers];
    _sizesOfLayers = sizesOfLayers;
    _layersTypes = layersTypes;
    _learningRate = lr;
    _currentBatchSize = batchSize;
    for (int i = 0; i < noOfLayers; i++) {
        if (i != 0) {
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i], _currentBatchSize);
        } else {
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], inputdim, i, _layersTypes[i], _currentBatchSize);
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


int Network::predictClass(int **inputIndices, float **inputValues, int *length, int *labels) {
    int correctPred = 0;
    for (int i = 0; i < _currentBatchSize; i++) {
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

int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int *labels, int iter) {

    float logloss = 0.0;
    int lowconf = 0;
    float prob = 0.0;
//# pragma omp parallel for
    int check = 0;
    float tmplr = _learningRate;
    if (ADAM) {
        tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
                (1 - pow(BETA1, iter + 1));
    }
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
                            _hiddenlayers[j - 1]->getAllNodes(), activenodesperlayer[j], sizes[j], tmplr, i);
                } else {
                    _hiddenlayers[j]->getNodebyID(activenodesperlayer[j + 1][k])->backPropagateFirstLayer(
                            inputIndices[i], inputValues[i], lengths[i], tmplr, i);
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
                std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
                tmp->_bias = tmp->_mirrorbias;
            }
        }
    }

    if(iter==1000000000) {
        float sum = 0;
//        for (int l=0; l<lengths[i]; l++){
//            sum+= inputValues[i][l];
//            cout<<inputValues[i][l]<<" ";
//        }
//        cout<<endl;
//        cout <<sum<<endl;
//        cout<<labels[i]<<endl;
//        cout << "Iteration " << i << endl;
        for (int j = 0; j < _numberOfLayers; j++) {
            cout << "Layer " << j << endl;
            for (int w = 0; w < _sizesOfLayers[j]; w++) {
                Node *tmp = _hiddenlayers[j]->getNodebyID(w);
                cout << "Weights for Node" << w << " ";
                for (int m = 0; m < tmp->_dim; m++) {
                    cout << tmp->_weights[m] << " ";
//                    cout<<tmp->_t[m]<<" ";
                }
                cout << endl;

                cout << "Bias " << tmp->_bias << endl;


            }
        }
        exit(0);
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
