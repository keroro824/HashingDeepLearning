#pragma once
#include "Layer.h"
#include <chrono>
#include "cnpy.h"
#include <sys/mman.h>

using namespace std;

class Network
{
private:
	Layer** _hiddenlayers;
	float _learningRate;
	int _numberOfLayers;
	int* _sizesOfLayers;
	NodeType* _layersTypes;
	float * _Sparsity;
	//int* _inputIDs;
	int  _currentBatchSize;


public:
	Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr);
	Layer* getLayer(int LayerID);
	int predictClass(int ** inputIndices, float ** inputValues, int * length, int ** labels, int *labelsize);
	int ProcessInput(int** inputIndices, float** inputValues, int* lengths, int ** label, int *labelsize, int iter, bool rehash, bool rebuild);
	void saveWeights(string file);
	~Network();
	void * operator new(size_t size){
            cout << "new Network size=" << size << endl;
            auto ret = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0);
            if (ret  == MAP_FAILED) {
               cout << "Fatal error of mmap(): " << strerror(errno) << endl;
               abort();
            }
            return ret;
        };
	void operator delete(void * pointer){munmap(pointer, sizeof(Network));};
};

