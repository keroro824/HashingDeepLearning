#pragma once
#include "Node.h"
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH.h"
#include "DensifiedWtaHash.h"
#include "cnpy.h"
#include <sys/mman.h>

using namespace std;

class Layer
{
private:
	NodeType _type;
	Node* _Nodes;
	int * _randNode;
	float* _normalizationConstants;
    int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;
    train* _train_array;


public:
	int _layerID, _noOfActive;
	size_t _noOfNodes;
	float* _weights;
	float* _adamAvgMom;
	float* _adamAvgVel;
	float* _bias;
	LSH *_hashTables;
	WtaHash *_wtaHasher;
    DensifiedMinhash *_MinHasher;
    SparseRandomProjection *_srp;
    DensifiedWtaHash *_dwtaHasher;
	int * _binids;
	Layer(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity, float* weights=NULL, float* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL);
	Node* getNodebyID(size_t nodeID);
	Node* getAllNodes();
	int getNodeCount();
	void addtoHashTable(float* weights, int length, float bias, int id);
	float getNomalizationConstant(int inputID);
	int queryActiveNodeandComputeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int queryActiveNodes(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeActivations(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeSoftmax(int** activenodesperlayer, float** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
	void saveWeights(string file);
	void updateTable();
	void updateRandomNodes();

	~Layer();

    void * operator new(size_t size){
        cout << "new Layer" << endl;
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED)
            std::cout << "mmap fail! No new layer!" << std::endl;
        return ptr;};
    void operator delete(void * pointer){munmap(pointer, sizeof(Layer));};

};
