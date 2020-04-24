#include "Network.h"
#include "Config.h"
#include <algorithm>
#include <iostream>
#include <math.h>
#include <omp.h>
#define DEBUG 1
using namespace std;

Network::Network(const std::vector<int> &sizesOfLayers,
                 const std::vector<NodeType> &layersTypes, int noOfLayers,
                 int batchSize, float lr, int inputdim,
                 const std::vector<int> &K, const std::vector<int> &L,
                 const std::vector<int> &RangePow,
                 const std::vector<float> &Sparsity, cnpy::npz_t arr)
    : _hiddenlayers(noOfLayers), _sizesOfLayers(sizesOfLayers),
      _layersTypes(layersTypes), _Sparsity(Sparsity),
      _numberOfLayers(noOfLayers), _learningRate(lr),
      _currentBatchSize(batchSize) {
  // Print("_Sparsity", _Sparsity);
  // Print("layersTypes", layersTypes);

  for (int i = 0; i < noOfLayers; i++) {
    int previousLayerNumOfNodes;
    if (i != 0) {
      previousLayerNumOfNodes = sizesOfLayers[i - 1];
    } else {
      previousLayerNumOfNodes = inputdim;
    }

    _hiddenlayers[i] =
        new Layer(sizesOfLayers[i], previousLayerNumOfNodes, i, _layersTypes[i],
                  _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i]);
  }
  cout << "after layer" << endl;
}

void Network::PrintNumberActive(const std::string &str) const {
  Layer &layer = *_hiddenlayers.back();
  cerr << str << "=";
  for (int i = 0; i < _currentBatchSize; i++) {
    size_t c = 0;
    for (size_t j = 0; j < layer.getAllNodes().size(); ++j) {
      // cerr << layer.getAllNodes()[j].getTrain(i)._ActiveinputIds << flush;
      if (layer.getAllNodes()[j].getTrain(i)._ActiveinputIds) {
        ++c;
      }
    }
    cerr << c << " ";
  }
  cerr << endl;
}

int Network::predictClass(Vec2d<int> &inputIndices, Vec2d<float> &inputValues,
                          const Vec2d<int> &labels) {
  int correctPred = 0;
  _hiddenlayers.back()->Reset();
  // PrintNumberActive("HH1");

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+ : correctPred) // num_threads(1)
  for (int i = 0; i < _currentBatchSize; i++) {
    Vec2d<int> activenodesperlayer(_numberOfLayers + 1);
    Vec2d<float> activeValuesperlayer(_numberOfLayers + 1);

    activenodesperlayer[0] = inputIndices[i];
    activeValuesperlayer[0] = inputValues[i];

    // inference
    for (int j = 0; j < _numberOfLayers; j++) {
      _hiddenlayers[j]->queryActiveNodeandComputeActivations(
          activenodesperlayer, activeValuesperlayer, i, labels[i],
          _Sparsity.at(_numberOfLayers + j), -1, false);
    }

    // compute softmax
    int noOfClasses = activenodesperlayer[_numberOfLayers].size();
    float max_act = -222222222;
    int predict_class = -1;
    for (int k = 0; k < noOfClasses; k++) {
      float cur_act = _hiddenlayers[_numberOfLayers - 1]
                          ->getNodebyID(activenodesperlayer[_numberOfLayers][k])
                          .getLastActivation(i);
      if (max_act < cur_act) {
        max_act = cur_act;
        predict_class = activenodesperlayer[_numberOfLayers][k];
      }
    }

    if (std::find(labels[i].begin(), labels[i].end(), predict_class) !=
        labels[i].end()) {
      correctPred++;
    }
  }
  // PrintNumberActive("HH2");

  auto t2 = std::chrono::high_resolution_clock::now();
  float timeDiffInMiliseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference takes " << timeDiffInMiliseconds / 1000
            << " milliseconds" << std::endl;

  return correctPred;
}

float Network::ProcessInput(Vec2d<int> &inputIndices, Vec2d<float> &inputValues,
                            const Vec2d<int> &labels, int iter, bool rehash,
                            bool rebuild) {
  float logloss = 0.0;
  std::vector<int> avg_retrieval(_numberOfLayers, 0);

  if (iter % 6946 == 6945) {
    //_learningRate *= 0.5;
    _hiddenlayers[1]->updateRandomNodes();
  }
  float tmplr = _learningRate;
  if (ADAM) {
    tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
            (1 - pow(BETA1, iter + 1));
  } else {
    //        tmplr *= pow(0.9, iter/10.0);
  }

  Vec3d<int> activeNodesPerBatch(_currentBatchSize); // batch, layer, node
  Vec3d<float> activeValuesPerBatch(_currentBatchSize);
#pragma omp parallel for //num_threads(1)
  for (int i = 0; i < _currentBatchSize; i++) {
    Vec2d<int> &activenodesperlayer = activeNodesPerBatch[i];
    activenodesperlayer.resize(_numberOfLayers + 1);

    Vec2d<float> &activeValuesperlayer = activeValuesPerBatch[i];
    activeValuesperlayer.resize(_numberOfLayers + 1);

    activenodesperlayer[0] =
        inputIndices[i]; // inputs parsed from training data file
    activeValuesperlayer[0] = inputValues[i];
    int in;
    // auto t1 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < _numberOfLayers; j++) {
      Layer &layer = *_hiddenlayers[j];
      // layer.Reset();

      in = _hiddenlayers[j]->queryActiveNodeandComputeActivations(
          activenodesperlayer, activeValuesperlayer, i, labels[i],
          _Sparsity.at(j), iter * _currentBatchSize + i, true);
      avg_retrieval[j] += in;
    }

    // Now backpropagate.
    // layers
    for (int j = _numberOfLayers - 1; j >= 0; j--) {
      Layer *layer = _hiddenlayers[j];
      Layer *prev_layer = _hiddenlayers[j - 1];

      // nodes
      for (int k = 0; k < activeNodesPerBatch[i][j + 1].size(); k++) {
        Node &node = layer->getNodebyID(activeNodesPerBatch[i][j + 1][k]);
        if (j == _numberOfLayers - 1) {
          // TODO: Compute Extra stats: labels[i];
          node.ComputeExtaStatsForSoftMax(layer->getNomalizationConstant(i), i,
                                          labels[i]);
        }
        if (j != 0) {
          node.backPropagate(prev_layer->getAllNodes(),
                             activeNodesPerBatch[i][j], activeNodesPerBatch[i][j].size(),
                             tmplr, i);
        } else {
          node.backPropagateFirstLayer(inputIndices[i], inputValues[i],
                                       inputIndices[i].size(), tmplr, i);
        }
      }
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  bool tmpRehash;
  bool tmpRebuild;

  for (int l = 0; l < _numberOfLayers; l++) {
    if (rehash & _Sparsity.at(l) < 1) {
      tmpRehash = true;
    } else {
      tmpRehash = false;
    }
    if (rebuild & _Sparsity.at(l) < 1) {
      tmpRebuild = true;
    } else {
      tmpRebuild = false;
    }
    if (tmpRehash) {
      _hiddenlayers[l]->hashTables().clear();
    }
    if (tmpRebuild) {
      _hiddenlayers[l]->updateTable();
    }
    int ratio = 1;
#pragma omp parallel for // num_threads(1)
    for (size_t m = 0; m < _hiddenlayers[l]->noOfNodes(); m++) {
      Node &tmp = _hiddenlayers[l]->getNodebyID(m);
      int dim = tmp.dim();
      std::vector<float> local_weights(dim);
      std::copy(tmp.weights().data(), tmp.weights().data() + dim,
                local_weights.begin());

      if (ADAM) {
        for (int d = 0; d < dim; d++) {
          float _t = tmp.getT(d);
          float Mom = tmp.adamAvgMom()[d];
          float Vel = tmp.adamAvgVel()[d];
          Mom = BETA1 * Mom + (1 - BETA1) * _t;
          Vel = BETA2 * Vel + (1 - BETA2) * _t * _t;
          local_weights[d] += ratio * tmplr * Mom / (sqrt(Vel) + EPS);
          tmp.adamAvgMom()[d] = Mom;
          tmp.adamAvgVel()[d] = Vel;
          tmp.setT(d, 0);
        }

        tmp.adamAvgMombias() =
            BETA1 * tmp.adamAvgMombias() + (1 - BETA1) * tmp.tbias();
        tmp.adamAvgVelbias() = BETA2 * tmp.adamAvgVelbias() +
                               (1 - BETA2) * tmp.tbias() * tmp.tbias();
        tmp.bias() += ratio * tmplr * tmp.adamAvgMombias() /
                      (sqrt(tmp.adamAvgVelbias()) + EPS);
        tmp.tbias() = 0;
      } else {
        std::copy(tmp.mirrorWeights(), tmp.mirrorWeights() + (tmp.dim()),
                  tmp.weights().data());
        tmp.bias() = tmp.mirrorbias();
      }
      if (tmpRehash) {
        std::vector<int> hashes;
        if (HashFunction == 1) {
          hashes = _hiddenlayers[l]->_wtaHasher->getHash(local_weights);
        } else if (HashFunction == 2) {
          hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(local_weights);
        } else if (HashFunction == 3) {
          hashes = _hiddenlayers[l]->_MinHasher->getHashEasy(
              _hiddenlayers[l]->binids(), local_weights, TOPK);
        } else if (HashFunction == 4) {
          hashes = _hiddenlayers[l]->_srp->getHash(local_weights);
        }

        std::vector<int> hashIndices =
            _hiddenlayers[l]->hashTables().hashesToIndex(hashes);
        _hiddenlayers[l]->hashTables().add(hashIndices, m + 1);
      }

      std::copy(local_weights.begin(), local_weights.end(),
                tmp.weights().data());
    }
  }

  if (DEBUG & rehash) {
    cout << "Avg sample size = " << avg_retrieval[0] * 1.0 / _currentBatchSize
         << " " << avg_retrieval[1] * 1.0 / _currentBatchSize << endl;
  }
  return logloss;
}

void Network::saveWeights(const std::string &file) {
  for (int i = 0; i < _numberOfLayers; i++) {
    _hiddenlayers[i]->saveWeights(file);
  }
}

Network::~Network() {
  for (int i = 0; i < _numberOfLayers; i++) {
    delete _hiddenlayers[i];
  }
}
