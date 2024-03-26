.. Copyright (C) 2024 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause
============
Contributing
============

Structure of the repository
===========================

The repository consists of several directories:

- benchmarks: Contains codes for default performance benchmarks for inference and training on artificial data.
- cmake: Contains files required for the cmake build system like common options and .cmake files to find packages.
- docs: Contains this documentation.
- dpcpp_bindings: Contains the Python - C++ bindings to use tiny-dpcpp-nn in Python.
- extern: Contains all extern dependencies either by providing the files directly (like doctest and json) or as submodules.
- include: All the header files for this project. Most of the code is contained in the headers since it heavily relies on template classes and functions.
- python: Python code. This will be removed in the future.
- source: Contains all the .cpp files of the tiny-dpcpp-nn code.
- test: Directory which contains all of the doctest unit-test.

Structure of tiny-dpcpp-nn/include
==================================

The most important parts of the code as of version 0.1 are contained in ``include/common``, ``include/encodings``, and ``include/network``.

The ``include/common`` directory contains small helper functions as well as the containers wrapping device memory, namely the classes ``DeviceMem``, ``DeviceMatrix``, and ``DeviceMatrices``, as well as the view classes ``DeviceMatrixView`` and ``DeviceMatricesView``. These latter two classes may be used in device kernels, whereas the classes ``DeviceMem``, ``DeviceMatrix``, and ``DeviceMatrices`` are only usable on the host.

The ``include/encodings`` directory contains all the available encodings, namely ``Identity``, ``SphericalHarmonics``, and ``Grid``. These encodings may be used to pad the input data to the required width before running inference or training.

Different loss functions are implemented in ``include/losses``. They are all derived from an abstract loss base class which is defined in ``include/losses/loss.h``.

The ``include/network`` directory contains the actual network implementation called ``SwiftNetMLP``. The relevant GPU kernels which facilitate the actual training and inference can be found in ``kernel_*`` files. As of version 0.1, we only support an esimd version which can be found in ``kernel_esimd.h``. The class ``network_with_encodings`` wraps a network and an encoding in a single object. At some point in the future, this should be consolidated.

Several optimization algorithms for training may be found in ``include/optimizers``. As of version 0.1, they are not yet implemented.