#include "Node.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
  Node::Node(size_t idx) 
    :_idx(idx)
  {
    //cerr << "Create Node" << endl;

  }

  Node::~Node() {

  }
}
