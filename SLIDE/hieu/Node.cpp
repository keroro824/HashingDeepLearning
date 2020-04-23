#include "Node.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
  Node::Node(size_t idx, SubVector<float> &nodeWeights, float &nodeBias)
    :_idx(idx)
    ,_nodeWeights(nodeWeights)
    ,_nodeBias(nodeBias)
  {
    //cerr << "Create Node" << endl;

  }

  Node::~Node() {

  }
}
