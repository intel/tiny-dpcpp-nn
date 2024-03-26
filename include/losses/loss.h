/**
 * @file loss.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of abstract loss base class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "DeviceMatrix.h"

template <typename T> class Loss {
  public:
    sycl::event evaluate(sycl::queue &q, const float loss_scale, const DeviceMatrixView<T> &predictions,
                         const DeviceMatrix<float> &targets, DeviceMatrix<float> &values, DeviceMatrix<T> &gradients) {
        SanityCheck(loss_scale, predictions, targets, values, gradients);

        return Kernel(q, predictions.m() * predictions.n(), loss_scale, predictions.GetPointer(), targets.data(),
                      values.data(), gradients.data());
    }

  protected:
    void SanityCheck(const float loss_scale, const DeviceMatrixView<T> &predictions, const DeviceMatrix<float> &targets,
                     DeviceMatrix<float> &values, DeviceMatrix<T> &gradients) {
        // Check if input dimensions match and if loss_scale is not 0
        const int n_elements = predictions.m() * predictions.n();
        assert(values.size() == n_elements);
        assert(gradients.size() == n_elements);
        assert(loss_scale != 0.0f);
        assert(targets.size() == n_elements);
    }

  protected:
    virtual sycl::event Kernel(sycl::queue &q, const size_t n_elements, const float loss_scale,
                               T const *const __restrict__ predictions, float const *const __restrict__ targets,
                               float *const __restrict__ values, T *const __restrict__ gradients) = 0;
};
