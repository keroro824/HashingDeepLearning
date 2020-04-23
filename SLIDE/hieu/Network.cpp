#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

namespace hieu {
Network::Network() 
{
  _layers.push_back(Layer());
  _layers.push_back(Layer());
  _layers.push_back(Layer());
}

Network::~Network() {}

} // namespace hieu