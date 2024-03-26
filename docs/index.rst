.. Copyright (C) 2023 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

tiny-dpcpp-nn
=============

The tiny-dpcpp-nn library is a library for the fast evaluation of inference and
training of neural networks in which each layer has the same, small width.
This implementation utilizes Intel's DPC++ and, in particular, the joint_matrix
extension to utilize Intel's AMX and XMX hardware to maximize performance.

License
-------

`BSD 3-Clause License <https://www.opensource.org/licenses/BSD-3-Clause>`_

.. Table of contents
.. -----------------

.. toctree::
  :includehidden:
  :maxdepth: 2

  manual/index
  api/index