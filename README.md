## Introduction

This repository implements a GPU-accelerated tiny neural network framework using Intel hardware, based on the [original CUDA implementation](https://github.com/NVlabs/tiny-cuda-nn). The implementation uses the Intel DPC++ compiler and relies on the SYCL language with optional ESIMD acceleration.

The network is optimized for loading both activation matrices and weights matrices into the GPU's fast L1 memory and registers. Computation of matrix multiplications is executed using Intel's `joint_matrix` extension, a high-level wrapper for systolic array operations.

## Performance
We benchmarked the thoughput of our network in training and inference on both [Intel Data Center GPU Max Series (Ponte Vecchio)](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html) and [Intel Arc Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) and compared our network with PyTorch.

To replicate the performance of the dpcpp code, please set `BUILD_BENCHMARK=ON` in `tiny-dpcpp-nn/CMakeLists.txt`, build `benchmark-all` and run the benchmark from the `build/` folder using

```
I_MPI_DEBUG=3 I_MPI_OFFLOAD=1 I_MPI_OFFLOAD_DOMAIN=[1,2] mpirun -n 2 ./benchmarks/benchmark-all
```

To replicate the performance of the pytorch code, please run
```
cd python/ && python benchmark_pytorch.py
```

Finally, plot the results using

```
python benchmarks/plot_results.py
```
### Performance on PVC
We reach up 60x to compared to PyTorch.

<table>
<tr>
<td>
<img src="benchmarks/results/training_throughput_comparison_width64.png" alt="Training Throughput Comparison" />
</td>
<td>
<img src="benchmarks/results/inference_throughput_comparison_width64.png" alt="Inference Throughput Comparison" />
</td>
</tr>
</table>

### Performance on Arc 770
We reach up to 20x compared to PyTorch.
![Image](data/performances.png)


## Features
- High-Performance Computing: Optimized to run efficiently on Intel Data Center GPUs, enabling high-throughput training and inference with up to 60x over PyTorch.
- Compatibility with PyTorch: Provides Python bindings that integrate seamlessly with the PyTorch ecosystem, enabling users to include GPU-accelerated MLPs in PyTorch applications.
- Versatile Neural Network Structures: Supports networks with multiple hidden layers and a variety of neuron configurations to fit different use cases and performance requirements.
- Multi-Resolution Hash Encoding: Includes implementation of [Multi-Resolution Hash Encoding](https://nvlabs.github.io/instant-ngp/), allowing the network to handle high-frequency features effectively.
- Cross-Platform Utilization: Designed to be run on various Intel GPUs, maximizing the portability and usability of the framework across different systems.

## Documentation

For a detailed documentation, please refer to [tiny-dpcpp-nn documentation](https://intel.github.io/tiny-dpcpp-nn/).

## Build

To build the tiny-nn librairy, you can clone the github repo on your machine and put your code in the source folder. After cloning, if you choose to use the pybindings, please recursive pull the pybind11 repositories via
```
git submodule update --init -- extern/pybind11
```

If you also want to pull the reference unittest data in `test/tiny_dpcpp_data`, which are ~500 MB of reference inputs, outputs, and weights, you can also run `git submodule update --init`. Note, that if `BUILD_REF_TEST=ON` in CMakeLists.txt, then `test/tiny_dpcpp_data` will be cloned as well.

Then you can build the library using :

```
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build/
cmake -D<options>=<ON/OFF> ..
make
```

where <options> are options that can be toggled on or off. See [Build Options](https://intel.github.io/tiny-dpcpp-nn/manual/build.html#build-options)

Note: To make the use of the network, you have to disable the implicit scaling on PVC which can be done by uncommenting the portion of the code indicated in the sample when creating the queue.


## Required Hardware and Software


## PyTorch extension
We provide a pybind wrapper of our tiny-dpcpp-nn implementation for seamless integration into PyTorch. Please refer to [tiny-dpcpp-nn pybind documentatino](https://intel.github.io/tiny-dpcpp-nn/manual/pytorch.html)

Install the latest [ipex](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) via

```
python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

## Tests
When setting the additional flag `BUILD_REF_TEST=ON`, additional data from [tiny-dpcpp-data](https://github.com/intel-sandbox/tiny-dpcpp-data) will be downloaded.

When setting the additional flag `BUILD_TORCH_TEST=ON`, the libtorch tests (`tnn_api.h`) will be built.

To have all tests, run:
```
cmake -DTARGET_DEVICE="PVC" -DBUILD_REF_TEST="ON" -DBUILD_REF_TEST="ON" ..
```

After all tests are build, you can run `bash test/run_tests.sh` to verfiy that the setup is correct. Please note that we provide tests for both the core `dpcpp` implementation and the `libtorch` wrapper implementation.

To test whether the pytorch bindings were installed correctly, please run
- `pytest python/tests/test_compare_torch_dpcpp.py` to ensure that forward and backward passes work properly.
- `python/tests/test_training_classification.py` and `python/tests/test_training_regression.py` to see if integration into PyTorch's optimiser works, and

## Acknowledgement
- The repository was developed and is maintained by Christoph Bauinger (christoph.bauinger@intel.com) and Kai Yuan (kai.yuan@intel.com).
- The original implementation of SwiftNet was conducted by Darius Dabert ([DariusDabert](https://github.com/DariusDabert)) and Adrien Tousnakhoff ([Tousnaaa](https://github.com/Tousnaaa))

## Citation

## Citation
If you found this work useful, please consider citing this work as:
```bibtex
@software{tiny-dpcpp-nn,
	author = {Bauinger, Christoph and Yuan, Kai},
	license = {BSD-3-Clause},
	month = {3},
	title = {{tiny-dpcpp-nn}},
	url = {https://github.com/intel/tiny-dpcpp-data/},
	version = {0.1},
	year = {2024}
}
```
