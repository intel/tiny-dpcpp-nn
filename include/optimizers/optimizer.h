/**
 * @file optimizer.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of an abstract Optimizer base class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <stdint.h>

#include "DeviceMem.h"

using bf16 = sycl::ext::oneapi::bfloat16;

class Optimizer {
  public:
    virtual ~Optimizer() {}

    virtual void step(queue q, float loss_scale, DeviceMem<bf16> &weights, DeviceMem<bf16> &weightsT,
                      DeviceMem<bf16> &gradients, int WIDTH) = 0;
};
