#include "Layer.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
  Layer::Layer() {
    cerr << "Create Layer" << endl;
    _nodes.push_back(Node());
    _nodes.push_back(Node());
    _nodes.push_back(Node());
    _nodes.push_back(Node());
    _nodes.push_back(Node());
  }

  Layer::~Layer() {

  }
}