#include "hieu-main.h"
#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <stdlib.h>

using namespace std;

namespace hieu {
int main(int argc, char *argv[]) {
  cerr << "Starting" << endl;
  size_t numEpochs = 5;

  Network mynet();

  for (size_t epoch = 0; epoch < numEpochs; epoch++) {
  }

  cerr << "Finished" << endl;
  exit(0);
}
} // namespace hieu
