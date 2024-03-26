.. Copyright (C) 2024 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

Benchmarks
==========

We include several benchmarks in the repository for the Inference and Training in the
``benchmark`` directory. Note that the benchmarks only include minimal tests to ensure correctness of the results.
Correctness tests are part of the unit tests, which can be found in the test directory.

Running the tests requires either a PVC GPU in the computer system (if built with ``-DTARGET_DEVICE="PVC"``)
or an ARC GPU (``-DTARGET_DEVICE="ARC"``).

The benchmark ``benchmarks/benchmark_all.cpp`` focuses on assessing the performance of training and inference across different use cases, such as NeRFs, PINNs, and Image Compression using tiny-dpcpp-nn. The C++ benchmark script comprises multiple configurations to run on specific batch sizes and iterations, involving MPI to distribute computational workloads efficiently.

In addition to general benchmarks, the code targets specialized use cases including:

- Large neural networks with a high count of hidden layers
- Neural networks tailored for image compression tasks
- Physics-Informed Neural Networks (PINNs), which incorporate domain knowledge on underlying physical laws into their structure
- Neural Radiance Fields (NeRF), that model the volumetric scene function for novel view synthesis

Examples
========

The examples located in the ``python/tests`` directory demonstrate usage scenarios and tests for validating the PyTorch extensions and functionalities for neural network training and inference with the ``tiny_dpcpp_nn_pybind`` module.

Gradient Descent Test Script
----------------------------

The gradient descent test script ``python/tests/test_train.py`` performs a simple check on gradient computations by comparing the gradients derived from two different model instantiations under multiple configurations. Utilizing features like custom loss functions and parameterized test inputs via PyTest, it validates against an assortment of model sizes, activation functions, and output functions.

.. code-block:: python

    # This script includes functions to perform gradient testing using parametrized configurations
    # with support for different layers, activation functions, and batch sizes.

    # The key components of this script include:
    # - CustomMSELoss: A sample MSE loss function implementation.
    # - train_model: The function that conducts the training process given a model and data.
    # - test_grad: A test function decorated with PyTest's parametrize decorator to test gradients.

Model Training and Evaluation Script
------------------------------------

This script ``python/tests/test_compare_torch_dpcpp.py`` showcases a model training process on a synthetic dataset with gradient checking. It highlights the setup of a DataLoader, loss function, optimizer choice, and embedding of a custom optimizer if needed. During training, the developed model attempts to approximate a simple linear function, and the results are visualized and compared with the ground truth.

.. code-block:: python

    # The key operations in this script include:
    # - Generating a random dataset.
    # - Setting up a DataLoader for batch processing.
    # - Training the network on the dataset using either Adam or a custom SGD optimizer.
    # - Reporting model parameter and output differences between models trained on different devices.
    # - Visualizing the loss over epochs and plotting the learned function against the true function.

Image Learning with PyTorch Extensions Script
---------------------------------------------

The script ``python/mlp_learning_an_image_pytorch.py`` demonstrates the adaptation of an image learning example using the PyTorch extension. It closely replicates the behavior of a CUDA sample and provides insights into how tiny-cuda-nn's PyTorch extension can be leveraged to train a multi-layer perceptron (MLP) to match an input image.

.. code-block:: python

    # The script performs operations such as:
    # - Using an Image loader class for image data handling.
    # - Setting up a model with input encoding and a network using tiny-cuda-nn's PyTorch extension.
    # - Optimization loop with Adam, along with saving intermediate output images at specified steps.
    # - Profiling the training runtime with performance metrics.

Each script provides a practical insight into using the PyTorch bindings with considerations for batch processing, parameter tuning, loss function choices, and utilizing accelerated compute resources. They also serve as tests to verify the correct integration and functioning of the PyTorch extensions.

Additionally, users interested in the Neural Radiance Fields (NeRF) application can refer to examples that use tiny-cuda-nn, showcasing how to accelerate NeRF training and inference with GPU support, e.g., NerfAcc.