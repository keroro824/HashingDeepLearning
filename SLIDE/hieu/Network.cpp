#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Network::Network(int) {
  size_t inputDim = 135909;

  cerr << "Create Network" << endl;
  _layers.push_back(new RELULayer(128, inputDim));
  _layers.push_back(new SoftmaxLayer(670091, 128));
}

Network::~Network() { cerr << "~Network" << endl; }

} // namespace hieu