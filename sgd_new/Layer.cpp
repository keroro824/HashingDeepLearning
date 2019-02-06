#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <climits>
#include "Config.h"
#include <bitset>
#include <fstream>

#include"cnpy.h"
#include<complex>
#include<cstdlib>
#include<string>

using namespace std;


Layer::Layer(int noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize,  int K, int L, int RangePow, float Sparsity, float* weights, float* bias)
{
    _layerID = layerID;
    _noOfNodes = noOfNodes;
    _Nodes = new Node*[noOfNodes];
    _type = type;
    _noOfActive = floor(_noOfNodes * Sparsity);
    _K = K;
    _L = L;

// create a list of random nodes just in case not enough nodes from hashtable for active nodes.
//	std::random_device rd;
//	std::mt19937 gen(rd());
//	std::uniform_int_distribution<> dis(0, _noOfNodes - 1);
    _randNode = new int[_noOfNodes];
    for (int n = 0; n < _noOfNodes; n++) {
        _randNode[n] = n;
    }

    std::random_shuffle ( _randNode, _randNode+_noOfNodes );

//TODO: Initialize Hash Tables and add the nodes. Done by Beidi
    _hashTables = new LSH(_K, _L, RangePow);

    if(HashFunction==1) {
        _wtaHasher = new WtaHash(_K * _L, previousLayerNumOfNodes);
    }else if (HashFunction==2) {
        _binids = new int[previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash(_K * _L, previousLayerNumOfNodes);
    }
    else if (HashFunction==3) {
        _binids = new int[previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash(_K * _L, previousLayerNumOfNodes);
        _MinHasher->getMap(previousLayerNumOfNodes, _binids);
    }else if (HashFunction==4){
        _srp = new SparseRandomProjection(previousLayerNumOfNodes, _K * _L, Ratio);
    }
    cout<<" Done initialize layer "<< _layerID <<" hashtables"<<endl;

    if (LOADWEIGHT) {
        _weights = weights;
        _bias = bias;
    }else{
        _weights = new float[_noOfNodes * previousLayerNumOfNodes]();
        _bias = new float[_noOfNodes];
        random_device rd;
        default_random_engine dre(rd());
        normal_distribution<float> distribution(0.0, 2.0/sqrt(previousLayerNumOfNodes*_noOfNodes));

        generate(_weights, _weights + _noOfNodes * previousLayerNumOfNodes, [&] () { return distribution(dre); });
        generate(_bias, _bias + _noOfNodes, [&] () { return distribution(dre); });
    }


#pragma omp parallel for
    for (size_t i = 0; i < noOfNodes; i++)
    {
        _Nodes[i] = new Node(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i, _bias[i]);
        _Nodes[i]->init(_hashTables, _wtaHasher, _MinHasher, _srp, _dwtaHasher, _binids, _L);
        addtoHashTable(_Nodes[i]->_weights, previousLayerNumOfNodes, _Nodes[i]->_bias, i);
    }

    if (type == NodeType::Softmax)
    {
        _normalizationConstants = new float[batchsize]();
        _inputIDs = new int[batchsize]();
    }
    cout<<" Done initialize layer "<< _layerID<<endl;

}

void Layer::addtoHashTable(float* weights, int length, float bias, int ID)
{
    //LSH logic
    int *hashes;
    if(HashFunction==1) {
        hashes = _wtaHasher->getHash(weights);
    }else if (HashFunction==2) {
        hashes = _dwtaHasher->getHashEasy(weights, length, TOPK);
    }else if (HashFunction==3) {
        hashes = _MinHasher->getHashEasy(_binids, weights, length, TOPK);
    }else if (HashFunction==4) {
        hashes = _srp->getHash(weights, length);
    }

    int * hashIndices = _hashTables->hashesToIndex(hashes);
    int * bucketIndices = _hashTables->add(hashIndices, ID+1);

    _Nodes[ID]->_indicesInTables = hashIndices;
    _Nodes[ID]->_indicesInBuckets = bucketIndices;

    delete [] hashes;

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


int Layer::queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int* label, int labelsize, float Sparsity, int iter)
{
    //LSH QueryLogic
    //TODO: it should return the active indices in indices and their activations in outvalues and outindices and outlength.
    //Now compute activations

    //Beidi. Query out all the candidate nodes
    int len;
    int in =0;
    if(Sparsity==1.0){
        len = _noOfNodes;
        lengths[layerIndex + 1] = len;
        activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
        for (int i = 0; i < len; i++)
        {
            activenodesperlayer[layerIndex + 1][i] = i;
        }
    }
    else {


        if (Mode==1) {
            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            int *hashes;
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
//                hashes = _srp->getHash(activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);

            // Get candidates from hashtable

            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        counts[label[i]] = _L;
                    }
                }
            }


            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }


        if(iter%20000==0) {
//        if (_type == NodeType::Softmax) {
////            cout << "-----------------layer " << layerIndex << "---------------------" << endl;
//////		cout << " Hashtable distribution" << endl;
//		    _hashTables->count();
            cout << " Number selected from hashIndicestable " << counts.size() << endl;
//            exit(0);
//        }
        }

            srand(time(NULL));
            int start = rand() % _noOfNodes;
            for (int i = start; i < _noOfNodes; i++) {
                if (counts.size() >= len) {
                    break;
                }
                if (counts.count(_randNode[i]) == 0) {
                    counts[_randNode[i]] = 0;
                }
            }


            if (counts.size() < len) {
                for (int i = 0; i < _noOfNodes; i++) {
                    if (counts.size() >= len) {
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

            std::random_shuffle(sortNodes.begin(), sortNodes.end());
            std::sort(begin(sortNodes), end(sortNodes));

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = sortNodes[i].second;
//                if ( activenodesperlayer[layerIndex + 1][i]== label){
//                    in = 1;
//                }
            }


            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

        }
        if (Mode==4) {
            int *hashes;
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
//                hashes = _srp->getHash(activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);

            // Get candidates from hashtable

            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        counts[label[i]] = _L;
                    }
                }
            }


            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            len = counts.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            in = len;

            int i=0;
            for (auto &&x : counts) {
                activenodesperlayer[layerIndex + 1][i] = x.first;
                i++;
            }




            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

        }
        else if (Mode == 2 & _type== NodeType::Softmax) {
            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            auto t1 = std::chrono::high_resolution_clock::now();
//            std::random_shuffle(_randNode, _randNode + _noOfNodes);
            bitset <MAPLEN> bs;
            int tmpsize = 0;
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        activenodesperlayer[layerIndex + 1][i] = label[i];
                        bs[label[i]] = 1;
                    }
                    tmpsize = labelsize;
                }
            }


//            for (int i = tmpsize; i < len; i++) {
            while(tmpsize<len){
                int v = rand() % _noOfNodes;
                if(bs[v]==0) {
                    activenodesperlayer[layerIndex + 1][tmpsize] = v;
                    bs[v]=1;
                    tmpsize++;
                }
            }



            auto t2 = std::chrono::high_resolution_clock::now();
            auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//            std::cout << "sampling "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

        }

        else if (Mode==3 & _type== NodeType::Softmax){

            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];
            vector<pair<float, int> > sortW;
            int what = 0;

            for (int s = 0; s < _noOfNodes; s++) {
                float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                         lengths[layerIndex], _Nodes[s]->_weights);
                tmp += _Nodes[s]->_bias;
//                cout<<iter<<" "<<tmp<<" ";
                if (find(label, label + labelsize, s) != label + labelsize) {
                    sortW.push_back(make_pair(-1000000000, s));
                    what++;
                }
                else{
                    sortW.push_back(make_pair(-tmp, s));
                }
            }
//            cout<<endl;

            if (what!=labelsize){
                cout<<labelsize<<" "<<what<<endl;
                cout<<label[0]<<" "<<_noOfNodes<<endl;
            }


            std::sort(begin(sortW), end(sortW));

//            ofstream outputFile("weights2",  std::ios_base::app);

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = sortW[i].second;
//                outputFile << sortW[i].second;

                if (find (label, label+labelsize, sortW[i].second)!= label+labelsize){
                    in=1;
                }
            }
//            outputFile << endl;
        }


        if (_layerID==10) {
            cout << "useNode ";
            for (int s = 0; s < len; s++) {
                int node_id = activenodesperlayer[layerIndex + 1][s];
                cout << node_id << " ";
            }
            cout << endl;

            float mean_compute = 0;
            for (int s = 1; s < len; s++) {
                int node_id = activenodesperlayer[layerIndex + 1][s];
                float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                         lengths[layerIndex], _Nodes[node_id]->_weights);
                tmp+=_Nodes[node_id]->_bias;
                mean_compute += tmp;
            }

            float total = 0;
            float overall = 0;

            vector<pair<float, int> > sortW;
            for (int s = 0; s < _noOfNodes; s++) {
                float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                         lengths[layerIndex], _Nodes[s]->_weights);
                tmp+=_Nodes[s]->_bias;
//            sortW.push_back(tmp);
                sortW.push_back(make_pair(tmp, s));
                overall += tmp;
            }
            std::sort(begin(sortW), end(sortW));

            int match = 0;
            for (int s = 0; s < len; s++) {
                total += sortW[_noOfNodes - 1 - s].first;
                for (int b = 0; b < len; b++) {
                    if (sortW[_noOfNodes - 1 - s].second == activenodesperlayer[layerIndex + 1][b]) {
                        match++;
                    }
                }
            }
            cout << "Matchedtopk= " << match * 1.0 / len << endl;
            cout << "hashingtopkmean= " << mean_compute / len << endl;
//			cout << "expected topk mean = " << total / len << endl;
            cout << "overallmean= " << overall / _noOfNodes << endl;

            if(label[0]==sortW[_noOfNodes-1].second){
                cout << "minmax= " << sortW[0].first<<" "<< sortW[_noOfNodes-2].first<<endl;
            }else{
                cout << "minmax= " << sortW[0].first<<" "<< sortW[_noOfNodes-1].first<<endl;
            }


        }


//        if (iter == 5000)
//        {
//
//                float mean_compute = 0;
//                int *hashes;
//                if (HashFunction==1) {
//                    hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
//                } else if(HashFunction==2){
//                    hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
//                } else if(HashFunction==3){
//                    hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
//                } else if (HashFunction==4) {
//                    hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
//                }
//                for (int s = 0; s < _noOfNodes; s++) {
//                    int node_id = s;
//                    float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
//                                             lengths[layerIndex], _Nodes[node_id]->_weights);
//                    int *table_hashes;
//                    if (HashFunction==1) {
//                        table_hashes = _wtaHasher->getHash(_Nodes[node_id]->_weights);
//                    }else if(HashFunction==2){
//                        table_hashes = _dwtaHasher->getHashEasy( _Nodes[node_id]->_weights, _Nodes[node_id]->_dim, TOPK);
//                    }else if(HashFunction==3){
//                        table_hashes = _MinHasher->getHashEasy(_binids, _Nodes[node_id]->_weights, _Nodes[node_id]->_dim, TOPK);
//                    }else if (HashFunction==4) {
//                        table_hashes = _srp->getHash(_Nodes[node_id]->_weights, _Nodes[node_id]->_dim);
//                    }
//
//
//                    float cp = collision(hashes, table_hashes, 1, _K*_L);
//                    cout <<"k=1 "<< tmp << " " << cp << endl;
//                    cp = collision(hashes, table_hashes, 2, _K*_L);
//                    cout <<"k=2 "<< tmp << " " << cp << endl;
//                    cp = collision(hashes, table_hashes, 3, _K*_L);
//                    cout <<"k=3 "<< tmp << " " << cp << endl;
//
//                }
////                exit(0);
//
//        }



    }



    //***********************************
    auto t1 = std::chrono::high_resolution_clock::now();
    activeValuesperlayer[layerIndex + 1] = new float[len]; //assuming its not initialized else memory leak;
    float maxValue = 0;
    if (_type == NodeType::Softmax)
        _normalizationConstants[inputID] = 0;

    int filtered = 0;
    for (int i = 0; i < len; i++)
    {
        activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]]->getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        if(activeValuesperlayer[layerIndex + 1][i]==0){
            filtered++;
        }
        if(_type == NodeType::Softmax){
            if (activeValuesperlayer[layerIndex + 1][i]>maxValue){
                maxValue = activeValuesperlayer[layerIndex + 1][i];
            }
        }
    }
//	cout <<"Layer "<<layerIndex<<" filtered "<<filtered*1.0/_noOfNodes<<endl;

    if(_type == NodeType::Softmax) {
        for (int i = 0; i < len; i++) {
            float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
            activeValuesperlayer[layerIndex + 1][i] =realActivation;
            _Nodes[activenodesperlayer[layerIndex + 1][i]]->_lastActivations[inputID] = realActivation;
            _normalizationConstants[inputID] += realActivation;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//            std::cout << "compute sm "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;


    return in;
}

void Layer::saveWeights(string file)
{
    if (_layerID==0) {
        cout<<"save for layer1"<<endl;
        cnpy::npz_save(file, "w_layer_0", _weights, {_noOfNodes, _Nodes[0]->_dim}, "w");
        cout<<"save for layer1"<<endl;
        cnpy::npz_save(file, "b_layer_0", _bias, {_noOfNodes}, "a");
        cout<<"save for layer1"<<endl;
    }else{
        cnpy::npz_save(file, "w_layer_"+ to_string(_layerID), _weights, {_noOfNodes, _Nodes[0]->_dim}, "a");
        cnpy::npz_save(file, "b_layer_"+ to_string(_layerID), _bias, {_noOfNodes}, "a");
    }
}

Layer::~Layer()
{

    for (size_t i = 0; i < _noOfNodes; i++)
    {
        free(_Nodes[i]);
        if (_type == NodeType::Softmax)
        {
            delete[] _normalizationConstants;
            delete[] _inputIDs;
        }
    }
    delete [] _Nodes;

    delete _wtaHasher;
    delete _dwtaHasher;
    delete _srp;
    delete _MinHasher;
}
