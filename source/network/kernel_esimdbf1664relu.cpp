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

using bf16 = sycl::ext::oneapi::bfloat16;

template class EsimdKernels<bf16, 64, 64, 64, Activation::ReLU, Activation::None>;
template class EsimdKernels<bf16, 64, 64, 64, Activation::ReLU, Activation::ReLU>;
template class EsimdKernels<bf16, 64, 64, 64, Activation::ReLU, Activation::Sigmoid>;

}
}
}