.. Copyright (C) 2024 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

====================
Building and linking
====================

Dependencies
============

Installing the
`Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html>`_
is sufficent to compile the library.

In particular, the project depends on:

- CMake >= 3.23
- Intel C++ compiler, icpx. Tested with version 2024.0
- Intel MPI
- Intel oneMKL
- Intel DPCT

Build from source using oneAPI
==============================

Clone the repository.

.. code:: console

    git clone PATH


Initialize the oneAPI environment.

.. code:: console

    . /opt/intel/oneapi/setvars.sh

Enter the directory containing your local copy of the repository (typically called tiny-dpcpp-nn) and run the following  cmake commands.

.. code:: console

    cmake -Bbuild
    cmake --build build


Build options
=============

The build allows customization with the following options.
To toggle them ON or OFF, set -D<option>=ON or OFF to cmake.

====================== ================================================== ========
Option                 Description                                        Default
====================== ================================================== ========
BUILD_DOCUMENTATION    Generate the documentation                         OFF
BUILD_BENCHMARK        Build benchmark executables                        ON
BUILD_EXAMPLE          Build examples                                     OFF
BUILD_TEST             Build unit tests                                   ON
BUILD_REF_TEST         Download reference data and build tests using it   OFF
BUILD_BWD_TEST         Compare our backward pass to an Eigen based code   OFF
BUILD_PYBIND           Build Python bindings for, e.g., PyTorch           OFF
====================== ================================================== ========

The following options enable further customization. To set them, 
use -D<option>=<value>

=============== ================================================== ============= 
Option          Description                                        Values
=============== ================================================== =============
TARGET_DEVICE   Build code either for "ARC" or "PVC"               "PVC", "ARC"
=============== ================================================== =============