#pragma once
#include <stddef.h>

namespace hieu {
  class Node {
  protected:
    size_t _idx;

  public:
    Node(size_t idx);
    virtual ~Node();

  };
}
