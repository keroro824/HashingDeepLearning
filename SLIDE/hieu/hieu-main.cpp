#include "hieu-main.h"
#include "../Util.h"
#include "Network.h"
#include <fstream>
#include <iostream>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

namespace hieu {
void EvalDataSVM(int numBatchesTest, Network &mynet, const std::string &path, size_t batchSize) {
  int totCorrect = 0;
  std::ifstream file(path);
  if (!file) {
    cout << "Error file not found: " << path << endl;
  }

  string str;
  // Skipe header
  std::getline(file, str);

  for (int i = 0; i < numBatchesTest; i++) {
    Vec2d<int> records(batchSize);
    Vec2d<float> values(batchSize);
    Vec2d<int> labels(batchSize);

    CreateData(file, records, values, labels, batchSize);

    int num_features = 0, num_labels = 0;
    for (int i = 0; i < batchSize; i++) {
      num_features += records[i].size();
      num_labels += labels[i].size();
    }

    std::cout << batchSize << " records, with " << num_features
      << " features and " << num_labels << " labels" << std::endl;
    size_t correctPredict = mynet.predictClass(records, values, labels);

  }
}

int main(int argc, char *argv[]) {
  cerr << "Starting" << endl;
  size_t numEpochs = 5;
  size_t batchSize = 128;
  size_t totRecords = 490449;
  size_t totRecordsTest = 153025;
  int numBatches = totRecords / batchSize;
  int numBatchesTest = totRecordsTest / batchSize;

  hieu::Network mynet;

  for (size_t epoch = 0; epoch < numEpochs; epoch++) {
    cerr << "epoch=" << epoch << endl;
    EvalDataSVM(20, mynet, "../dataset/Amazon/amazon_test.txt", batchSize);

  }

  cerr << "Finished" << endl;
  exit(0);
}
} // namespace hieu
