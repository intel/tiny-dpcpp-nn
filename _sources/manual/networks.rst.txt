.. Copyright (C) 2024 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause


========
Network
========

Tiny-dpcpp-nn is an implementation of multi-layer perceptrions (MLPs) for Intel GPUs.
In particular, this implementation focuses on MLPs with constant, small power of 2
network widths (16, 32, 64, 128) and aims to maximize the performance of inference and
training by fusing the requisite matrix operations in the separate layers into a single
operation. 


Multi-Layer Perceptrons
=======================

Fused MLPs
==========

As indicate above, Fused MLPs fuse the matrix operations of several layers into a single 
operation. In general, this decreases the number of global memory access and thus increases
the arithmetic intensity and performance. 
Especially Inference benefits from this since the load of the input of each layer and the store 
of the output of each layer can be reduced to a single load of the input to the network 
and a single store of the output of the network.

ESIMD Kernels
=============

The fused forward pass and the backward pass are implemented in Intel's ESIMD SYCL extension.
They can be found in include/network/kernel_esimd.h in the functions forward_impl_general and
backward_impl, respectively. The forward_impl_general is templated and re-used for the forward
pass required as part of the training of the network as well as the Inference.
