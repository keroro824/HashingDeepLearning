# SLIDE

The SLIDE package contains the source code for reproducing the main experiments in this [paper](https://arxiv.org/abs/1903.03129).

## Dataset

The Datasets can be downloaded in [Amazon-670K](https://drive.google.com/open?id=0B3lPMIHmG6vGdUJwRzltS1dvUVk).

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

Firstly,  [CNPY](https://github.com/rogersce/cnpy) package needs to be installed.

Additionally, Transparent Huge Pages must be enabled.  SLIDE requires approximately 900 2MB pages, and 10 1GB pages.


Please see the [Instructions](https://wiki.debian.org/Hugepages) to enable Hugepages on Ubuntu.
([Instructions](https://wiki.debian.org/Hugepages)). Please note that only Skylake or newer architectures support Hugepages. For older Haswell processors, we need to remove the flag `-mavx512f` from the `OPT_FLAGS` line in Makefile. You can also revert to the commit `2d10d46b5f6f1eda5d19f27038a596446fc17cee` to ignore the HugePages optmization and still use SLIDE (which could lead to a 30% slower performance). 



Run

```make```

```./runme Config_amz.csv```

Note that `Makefile` needs to be modified based on the CNPY path. Also the `trainData, testData, logFile` in Config_amz.csv needs to be changed accordingly too.


