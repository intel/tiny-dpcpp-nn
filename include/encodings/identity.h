/**
 * @file identity.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementaiton of identity encoding class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "DeviceMatrix.h"
#include "common.h"
#include "common_device.h"
#include "encoding.h"
#include <stdint.h>

#include <numeric>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

template <typename T> class IdentityEncoding : public Encoding<T> {
  public:
    IdentityEncoding(uint32_t n_dims_to_encode, const float scale = 1.0f, const float offset = 0.0f)
        : m_n_dims_to_encode{n_dims_to_encode}, m_scale{scale}, m_offset{offset}, m_n_to_pad{0} {}

    std::unique_ptr<Context> forward_impl(sycl::queue *const q, const DeviceMatrixView<float> input,
                                          DeviceMatrixView<T> *output = nullptr, bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {
        const uint32_t loc_padded_output_width = padded_output_width();

        if (!output || loc_padded_output_width == 0) return std::make_unique<Context>();
        if (input.n() != m_n_dims_to_encode)
            throw std::invalid_argument("input dimensions do not coincide with encoder");
        if (output->m() != input.m()) throw std::invalid_argument("Differing row numbers");
        if (output->n() != loc_padded_output_width)
            throw std::invalid_argument("number of cols has to be padded output width.");

        const uint32_t n_elements = input.m() * loc_padded_output_width;
        if (n_elements <= 0) return std::make_unique<Context>();

        float const *const loc_in = input.GetPointer();
        T *const loc_out = output->GetPointer();

        auto loc_n_dims_to_encode = m_n_dims_to_encode;
        auto loc_scale = m_scale;
        auto loc_offset = m_offset;

        // manually, because we dont have MatrixView on device
        auto unpadded_stride = input.n();

        // Create a command group to issue commands to the queue
        q->parallel_for(n_elements, [=](id<1> index) {
             const uint32_t encoded_index = index;

             // columns which are batch size
             const uint32_t i = encoded_index / loc_padded_output_width;
             const uint32_t j = encoded_index - i * loc_padded_output_width;

             const uint32_t idx = i * loc_padded_output_width + j;
             const uint32_t unpadded_idx = i * unpadded_stride + j;

             if (j >= loc_n_dims_to_encode)
                 loc_out[idx] = (T)1;
             else
                 loc_out[idx] = loc_in[unpadded_idx] * loc_scale + loc_offset;
         }).wait();
        return std::make_unique<Context>();
    }

    void backward_impl(sycl::queue *const q, const Context &ctx, const DeviceMatrixView<float> input,
                       const DeviceMatrixView<T> dL_doutput, DeviceMatrixView<T> *gradients = nullptr,
                       DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        throw std::logic_error("Not yet implemented.");

        if (!dL_dinput) return;

        const uint32_t n_elements = input.n() * m_n_dims_to_encode;
        if (n_elements <= 0) return; // nothing to do

        float *const dL_dx = dL_dinput->data();
        T const *const dL_dy = dL_doutput.GetPointer();
        auto loc_scale = m_scale;

        q->parallel_for(n_elements, [=](auto item) {
            const uint32_t idx = item;
            dL_dx[idx] = (float)dL_dy[idx] * loc_scale;
        });
    }

    uint32_t input_width() const override { return m_n_dims_to_encode; }

    uint32_t padded_output_width() const override { return m_n_dims_to_encode + m_n_to_pad; }

    uint32_t output_width() const override { return padded_output_width(); }

    void set_padded_output_width(const uint32_t padded_output_width) override {
        if (padded_output_width < m_n_dims_to_encode)
            throw std::invalid_argument("Padded width has to be larger than unpadded. m_n_dims_to_encode: " +
                                        std::to_string(m_n_dims_to_encode) +
                                        ", padded_output_width: " + std::to_string(padded_output_width));

        m_n_to_pad = padded_output_width - m_n_dims_to_encode;
    }

    void initialize_params(float *params_full_precision, float scale = 1) override {}

  private:
    /// number of elements before padding in padding dimension
    ///(cols)
    uint32_t m_n_dims_to_encode;

    float m_scale;
    float m_offset;

    // derived sizes
    uint32_t m_n_to_pad;
};