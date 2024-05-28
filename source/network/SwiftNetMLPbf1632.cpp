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

#include "SwiftNetMLP.h"

#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

template class SwiftNetMLP<bf16, 32>;