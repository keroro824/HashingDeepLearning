# SLIDE

The SLIDE package contains the source code for reproducing the main experiments in this [paper](https://arxiv.org/abs/1903.03129).

For Optimized Code on CPUs (with AVX, BFloat and other memory optimization) from the newer [paper](https://proceedings.mlsys.org/paper/2021/file/3636638817772e42b59d74cff571fbb3-Paper.pdf) please refer [here](https://github.com/RUSH-LAB/SLIDE) 

## Dataset

The Datasets can be downloaded in [Amazon-670K](https://drive.google.com/open?id=0B3lPMIHmG6vGdUJwRzltS1dvUVk). Note that the data is sorted by labels so please shuffle at least the validation/testing data.

## TensorFlow Baselines

We suggest directly get TensorFlow docker image to install [TensorFlow-GPU](https://www.tensorflow.org/install/docker).
For TensorFlow-CPU compiled with AVX2, we recommend using this precompiled [build](https://github.com/lakshayg/tensorflow-build).

Also there is a TensorFlow docker image specifically built for CPUs with AVX-512 instructions, to get it use:

```bash
docker pull clearlinux/stacks-dlrs_2-mkl    
```

`config.py` controls the parameters of TensorFlow training like `learning rate`. `example_full_softmax.py, example_sampled_softmax.py` are example files for `Amazon-670K` dataset with full softmax and sampled softmax respectively.

Run

```bash
python python_examples/example_full_softmax.py
python python_examples/example_sampled_softmax.py
```

## Running SLIDE

### Dependencies

- CMake v3.0 and above
- C++11 Compliant compiler
- Linux: Ubuntu 16.04 and newer
- Transparent Huge Pages must be enabled.
  - SLIDE requires approximately 900 2MB pages, and 10 1GB pages: ([Instructions](https://wiki.debian.org/Hugepages))

### Notes:

- For simplicity, please refer to the our [Docker](https://hub.docker.com/repository/docker/ottovonxu/slide) image with all environments installed. To replicate the experiment without setting Hugepages, please download [Amazon-670K](https://drive.google.com/open?id=0B3lPMIHmG6vGdUJwRzltS1dvUVk) in path ```/home/code/HashingDeepLearning/dataset/Amazon``` 

- Also, note that only Skylake or newer architectures support Hugepages. For older Haswell processors, we need to remove the flag `-mavx512f` from the `OPT_FLAGS` line in Makefile. You can also revert to the commit `2d10d46b5f6f1eda5d19f27038a596446fc17cee` to ignore the HugePages optimization and still use SLIDE (which could lead to a 30% slower performance). 

- This version builds all dependencies (which currently are [ZLIB](https://github.com/madler/zlib/tree/v1.2.11) and [CNPY](https://github.com/sarthakpati/cnpy)).

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
