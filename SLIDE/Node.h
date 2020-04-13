#pragma once
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <linux/mman.h>
#include <sys/mman.h>
#include <asm-generic/mman-common.h>
#include "Util.h"

using namespace std;

enum NodeType
{ ReLU, Softmax};

struct train {
    float _lastDeltaforBPs;
    float _lastActivations;
    float _lastGradients;
    int _ActiveinputIds;

    void * operator new(size_t size){
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED){
            std::cout << "mmap failed at train." << std::endl;
        }
        return ptr;
    }
    void* operator new (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new (std::size_t size, void* ptr){return operator new (size);};
    void* operator new[] (std::size_t size){
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED){
            std::cout << "mmap fail! No train array!" << std::endl;
        }
        return ptr;
    }
    void* operator new[] (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new[] (std::size_t size, void* ptr){return operator new (size);};

    void operator delete(void * ptr){munmap(ptr, sizeof(train));};
    void operator delete (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(train));};
    void operator delete (void* ptr, void* voidptr2){};
    // TODO: The size to be munmap'd should be the entire array, not just a single object
    void operator delete[](void * ptr){munmap(ptr, sizeof(train));};
    void operator delete[] (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(train));};
    void operator delete[] (void* ptr, void* voidptr2){};

} __attribute__ ((aligned (64)));

class Node
{
private:
	int _activeInputs;
  NodeType _type;
  SubVector<train> _train;
  int _currentBatchsize;
  size_t _layerNum, _IDinLayer;
  size_t _dim;
  int* _indicesInTables;
  int* _indicesInBuckets;
  SubVector<float> _weights;
  float* _mirrorWeights;

  SubVector<float> _adamAvgMom;
  SubVector<float> _adamAvgVel;
  float _adamAvgMombias = 0;
  float _adamAvgVelbias = 0;

  float _bias = 0;
  float _tbias = 0;
  float _mirrorbias = 0;

  std::vector<float> _t; //for adam

public:

  const size_t &dim() const { return _dim; }
  int *&indicesInTables() { return _indicesInTables; }
  int *&indicesInBuckets() { return _indicesInBuckets; }
  
  const SubVector<float> &weights() const { return _weights; }
  SubVector<float> &weights() { return _weights; }

  const float *mirrorWeights() const { return _mirrorWeights; } // not adam

  float *adamAvgMom() { return _adamAvgMom.data(); }
  float *adamAvgVel() { return _adamAvgVel.data(); }
  float &adamAvgMombias() { return _adamAvgMombias; }
  float &adamAvgVelbias() { return _adamAvgVelbias; }

  float &bias() { return _bias; }
  float &tbias() { return _tbias; }
  const float &mirrorbias() const { return _mirrorbias;  }

  float getT(size_t idx) const { return _t[idx]; }
  void setT(size_t idx, float val) { _t[idx] = val; }

  ////////////////////
	Node(){};
	void Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, std::vector<float> &allWeights, float bias, std::vector<float> &allAdamAvgMom, std::vector<float> &allAdamAvgVel, std::vector<train> &train_blob);
	float getLastActivation(int inputID) const;
	void incrementDelta(int inputID, float incrementValue);
	float getActivation(const std::vector<int> &indices, const std::vector<float> &values, int length, int inputID);
	bool getInputActive(int inputID);
	bool getActiveInputs(void);
	void SetlastActivation(int inputID, float realActivation);
	void ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, const std::vector<int> &label, int labelsize);
	void backPropagate(std::vector<Node> &previousNodes, const std::vector<int> &previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID);
	void backPropagateFirstLayer(const std::vector<int> &nnzindices, const std::vector<float> &nnzvalues, int nnzSize, float learningRate, int inputID);
	~Node();

    void * operator new(size_t size){
        std::cout << "new Node" << std::endl;
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED){
            std::cout << "mmap failed at Node." << std::endl;
        }
        return ptr;
    }
    void* operator new (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new (std::size_t size, void* ptr){return operator new (size);};
    void* operator new[] (std::size_t size){
        std::cout << "new Node array" << std::endl;
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED){
            std::cout << "mmap failed at Node array." << std::endl;
        }
        return ptr;
    }
    void* operator new[] (std::size_t size, const std::nothrow_t& nothrow_value){return operator new (size);};
    void* operator new[] (std::size_t size, void* ptr){return operator new (size);};

    void operator delete(void * ptr){munmap(ptr, sizeof(Node));};
    void operator delete (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(Node));};
    void operator delete (void* ptr, void* voidptr2){};
    // TODO: should munmap the size of the entire array, not a single Node
    void operator delete[](void * ptr){munmap(ptr, sizeof(Node));};
    void operator delete[] (void* ptr, const std::nothrow_t& nothrow_constant){munmap(ptr, sizeof(Node));};
    void operator delete[] (void* ptr, void* voidptr2){};

	//only for debugging
	float purturbWeight(int weightid, float delta);
	float getGradient(int weightid, int inputID, float InputVal);
};
