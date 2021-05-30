# softmax

This is a command line program performing [softmax function](https://en.wikipedia.org/wiki/Softmax_function) on it's arguments.

**This software has only been built on Nvidia Jetson TK1**

## Pre requisites

* [CUDA](https://developer.nvidia.com/cuda-downloads)
* Linux
* c++ compiler. Tested on gcc.
 
This software has only been tested on [Nvidia Jetson TK1].

A [cmake](https://cmake.org) setup is also provided. However this is setup has not been tested.

## Install

Building on jetson TK1

```bash
make -j 4
```

## Usage

This program supports two modes; Calculcating softmax based on numbers from command line arguments or calculcating softmax based on numbers from stdin.

Calculcating softmax based on numbers from command line arguments can be performed  as shown below.

```bash
$ ./softmax 1.0 1.0 1.0
0.333 0.333 0.333
```

Calculcating softmax based on numbers from stdin can be performed as shown below.

```bash
$ echo "1.0 1.0 1.0" | ./softmax
0.333 0.333 0.333
```

### Command line flags

These command line flags are supported.

| Flag       | Description                                  |
| :--------- |:-------------------------------------------- |
| -v         | Verbose logging                              |
| -h         | Prints the help message                      |
| --cpu      | Run the softmax calculation on the CPU       |
| --device=0 | Run the computation on the first GPU device  |
| --device=1 | Run the computation on the second GPU device |

## Limitations and Issues

For **n > 2048** the program will fall back to calculate the function on the CPU. This is limited by the number of threads that a Nvidia GPU can start on a single kernel launch.

[Makefile](Makefile) holds the compilation process. This file has been hard coded to compile for CUDA capability 3.2. For commpilation on your device update your compute capability in this file. Update  the code block `-gencode arch=compute_32,code=sm_32`. For more info see [Nvidia Developer Zone - CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

For large amount of input numbers the denominator of softmax will overflow. There is simply not enough information in IEEE 754 floating point to store this denominator. In practice this limit is reached between **n = 750** and **n = 800** for the artimetic series 1, 2, 3, 4, ... n. This is easily spotted by detecting `-nan` (not a number) for all output numbers.

Example `echo {1..800} | ./softmax`

## Appendix

* [Developer Tips](developer-tips.md)
* [6_Advanced/reduction from CUDA 6.5 samples](https://docs.nvidia.com/cuda/cuda-samples/index.html) was the starting point for the cuda kernels.
* [Optimizing Parallel Reduction in CUDA, Mark Harris, NVIDIA Developer Technology](http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) explains the CUDA kernels tips and techniques.
