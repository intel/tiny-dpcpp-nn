/**
 * @file SwiftNetMLP.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief template instantion to reduce compile time
 * @version 0.1
 * @date 2024-03-21
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "kernel_esimd.h"

#include <sycl/sycl.hpp>

namespace tinydpcppnn {
namespace kernels {
namespace esimd {

template class EsimdKernels<sycl::half, 64, 64, 64, Activation::None, Activation::None>;
template class EsimdKernels<sycl::half, 64, 64, 64, Activation::None, Activation::ReLU>;
template class EsimdKernels<sycl::half, 64, 64, 64, Activation::None, Activation::Sigmoid>;

}
}
}