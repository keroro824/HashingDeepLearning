#pragma once
#include "Node.h"
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH.h"
#include "DensifiedWtaHash.h"
#include "cnpy.h"
#include <sys/mman.h>
#include <vector>

class Layer
{
private:
	  NodeType _type;
    std::vector<Node> _Nodes;
    std::vector<int> _randNode;
    std::vector<float> _normalizationConstants;
    int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;
    std::vector<train> _train_array;

    int _layerID, _noOfActive;
    size_t _noOfNodes;
    std::vector<float> _weights;
    std::vector<float> _adamAvgMom;
    std::vector<float> _adamAvgVel;
    std::vector<float> _bias;
    std::vector<int> _binids;
    LSH _hashTables;

public:
  WtaHash *_wtaHasher;
  DensifiedMinhash *_MinHasher;
  SparseRandomProjection *_srp;
  DensifiedWtaHash *_dwtaHasher;

  LSH & hashTables() { return _hashTables; }
  size_t noOfNodes() const { return _noOfNodes; }
  const std::vector<int> &binids() const { return _binids; }

  Layer(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity);
	Node &getNodebyID(size_t nodeID);
  std::vector<Node> &getAllNodes();
	void addtoHashTable(SubVector<float> &weights, float bias, int id);
	float getNomalizationConstant(int inputID) const;
	int queryActiveNodeandComputeActivations(Vec2d<int> &activenodesperlayer, Vec2d<float> &activeValuesperlayer, std::vector<int> &inlenght, int layerID, int inputID,  const std::vector<int> &label, int labelsize, float Sparsity, int iter);
	void saveWeights(const std::string &file);
	void updateTable();
	void updateRandomNodes();

	~Layer();

    void * operator new(size_t size){
      std::cout << "new Layer" << std::endl;
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
