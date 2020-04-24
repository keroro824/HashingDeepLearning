#pragma once
#include "DensifiedMinhash.h"
#include "DensifiedWtaHash.h"
#include "LSH.h"
#include "Node.h"
#include "WtaHash.h"
#include "cnpy.h"
#include "srp.h"
#include <vector>

class Layer {
private:
  NodeType _type;
  std::vector<Node> _Nodes;
  std::vector<int> _randNode;
  std::vector<float> _normalizationConstants;
  int _K, _L, _RangeRow, _previousLayerNumOfNodes, _batchsize;

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

  LSH &hashTables() { return _hashTables; }
  size_t noOfNodes() const { return _noOfNodes; }
  const std::vector<int> &binids() const { return _binids; }

  Layer(size_t noOfNodes, int previousLayerNumOfNodes, int layerID,
        NodeType type, int batchsize, int K, int L, int RangePow,
        float Sparsity);
  Node &getNodebyID(size_t nodeID);
  std::vector<Node> &getAllNodes();
  void addtoHashTable(SubVector<float> &weights, float bias, int id);
  float getNomalizationConstant(int inputID) const;
  int queryActiveNodeandComputeActivations(
      Vec2d<int> &activenodesperlayer, Vec2d<float> &activeValuesperlayer, int inputID, const std::vector<int> &label,
      float Sparsity, int iter, bool train);
  void saveWeights(const std::string &file);
  void updateTable();
  void updateRandomNodes();
  void Reset();

  ~Layer();
};
