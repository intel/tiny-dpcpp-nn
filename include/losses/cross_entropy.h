/**
 * @file cross_entropy.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of cross entropy loss class.
 * TODO: actually implement it.
 * @version 0.1
 * @date 2024-01-19
 * 
 * Copyright (c) 2024 Intel Corporation
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "loss.h"

template <typename T> class CrossEntropyLoss : public Loss<T> {
  protected:
    void Kernel(const sycl::queue &q, const size_t n_elements, const float loss_scale,
                T const *const __restrict__ predictions, float const *const __restrict__ targets,
                float *const __restrict__ values, T *const __restrict__ gradients) override {
        throw std::invalid_argument("CrossEntropuLoss not yet implemented.");
    }
};
