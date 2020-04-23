#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
Network::Network(int) 
{
  cerr << "Create Network" << endl;
  _layers.push_back(new RELULayer(128));
  _layers.push_back(new SoftmaxLayer(670091));
}

Network::~Network() 
{
  cerr << "~Network" << endl;
}

} // namespace hieu