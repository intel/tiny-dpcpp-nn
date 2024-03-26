.. Copyright (C) 2024 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

=====================================
tiny_dpcpp_nn_pybind Module Documentation
=====================================

This documentation provides guidance for the usage and development of the
``tiny_dpcpp_nn_pybind`` module. This Python module is built with pybind11 and
CMake, offering Python bindings for a C++ extension. It is built using `IPEX' PyTorch C++ Library <https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/examples.html#c>`_.

Installation
------------

To build and install the ``tiny_dpcpp_nn_pybind`` module, ensure that you have
Python 3.9 or 3.10 installed on your system. It is recommended to use a Conda
environment to manage the dependencies and Python versions.

Using Conda
~~~~~~~~~~~

Install Conda if it is not already installed, and create a new environment with
Python 3.9 or 3.10:

.. code-block:: sh

    conda create -n tinydpcpp_nn python=3.10
    conda activate tinydpcpp_nn

With the Conda environment activated, you can proceed to install the module.

Installing Intel Extension for PyTorch (IPEX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For XPU related operations, Intel Extension for PyTorch (IPEX) is required. Run the following command:

.. code-block:: sh

    python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel-extension-for-pytorch==2.0.120+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/

Make sure the Conda environment is active when you install IPEX to ensure it is
available within the environment. Other versions and further installation details can be found in the `IPEX documentation <https://intel.github.io/intel-extension-for-pytorch/index.html#installation>`_.

Installing the Module
~~~~~~~~~~~~~~~~~~~~~

Clone the repository and navigate to the root directory of the module. Then, install
the module in editable mode using pip:

.. code-block:: sh

    cd dpcpp_bindings
    pip install -e .

Usage
-----

After installation, the ``tiny_dpcpp_nn_pybind`` module can be imported into your
Python projects and scripts. For convenience, a wrapper module is provided named ``tiny_dpcpp_nn`` (see ``dpcpp_bindings/modules.py``).

Example Usage
~~~~~~~~~~~~~

An example of creating a model with the module might look as follows:

.. code-block:: python

    import tiny_dpcpp_nn as tcnn
    import torch

    n_input_dims = 3   # Example number of input dimensions
    n_output_dims = 2  # Example number of output dimensions

    config = {
            "otype": "Identity",
            "n_dims_to_encode": n_input_dims,
            "scale": 1.0,
            "offset": 0.0,
            }

    # Option 1: efficient Encoding+Network combo.
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims, n_output_dims,
        config["encoding"], config["network"]
    )

    # Option 2: separate modules. Slower but more flexible.
    encoding = tcnn.Encoding(n_input_dims, config["encoding"])
    network = tcnn.Network(encoding.n_output_dims, n_output_dims, config["network"])
    model = torch.nn.Sequential(encoding, network)

Refer to sample usage patterns and configurations provided in the bundled
examples, such as ``samples/mlp_learning_an_image_pytorch.py``.