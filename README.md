# SLIDE

The SLIDE package contains the source code for reproducing the main experiments in this [paper](https://arxiv.org/abs/1903.03129).

## Dataset

The Datasets can be downloaded in [Amazon-670K](https://drive.google.com/open?id=0B3lPMIHmG6vGdUJwRzltS1dvUVk).

## Tensorflow Baselines

We suggest directly get Tensorflow docker image to install [Tensorflow-GPU] (https://www.tensorflow.org/install/docker).
For Tensorflow-CPU compiled with AVX2, we recommend using this precompiled [build](https://github.com/lakshayg/tensorflow-build).

`config.py` controls the parameters of Tensorflow training like `learning rate`. `example_full_softmax.py, example_sampled_softmax.py` are example files for `Amazon-670K` dataset with full softmax and sampled softmax respectively.

Run

```python python_examples/example_full_softmax.py```
``` python python_examples/example_sampled_softmax.py```

## Running SLIDE

### Dependencies

- CMake v3.0 and above
- C++11 Compliant compiler
- Linux: Ubuntu 16.04 and newer
- Transparent Huge Pages must be enabled.
  - SLIDE requires approximately 900 2MB pages, and 10 1GB pages: ([Instructions](https://wiki.debian.org/Hugepages))

This version builds all dependencies (which currently are [ZLIB](https://github.com/madler/zlib/tree/v1.2.11) and [CNPY](https://github.com/sarthakpati/cnpy)).

### Commands

Change the paths in ```./SLIDE/Config_amz.csv``` appropriately.

```bash
git clone https://github.com/sarthakpati/HashingDeepLearning.git
cd HashingDeepLearning
mkdir bin
cd bin
cmake ..
make
./runme ../SLIDE/Config_amz.csv
```
