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
    IdentityEncoding(const uint32_t n_dims_to_encode, const uint32_t padded_output_width, 
        const float scale, const float offset, sycl::queue& Q)
        : Encoding<T>(n_dims_to_encode, n_dims_to_encode, padded_output_width, Q), 
            m_scale{scale}, m_offset{offset} {}

    std::unique_ptr<Context> forward_impl(const DeviceMatrixView<float> input,
                                          DeviceMatrixView<T> *output = nullptr, bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {

        if (!output || this->get_padded_output_width() == 0) return std::make_unique<Context>();
        if (input.n() != this->get_input_width())
            throw std::invalid_argument("input dimensions do not coincide with encoder");
        if (use_inference_params) throw std::invalid_argument("Cannot yet use inference params");
        if (output->m() != input.m()) throw std::invalid_argument("Differing row numbers");
        if (output->n() != this->get_padded_output_width())
            throw std::invalid_argument("number of cols has to be padded output width.");

        const size_t n_elements = input.m() * this->get_padded_output_width();
        if (n_elements == 0) return std::make_unique<Context>();

        float const *const loc_in = input.GetPointer();
        T *const loc_out = output->GetPointer();

        auto loc_n_dims_to_encode = this->get_input_width();
        auto loc_scale = m_scale;
        auto loc_offset = m_offset;

        // manually, because we dont have MatrixView on device
        auto loc_input_stride = input.n();
        auto loc_padded_output_width = this->get_padded_output_width();

        //
        this->get_queue().parallel_for(n_elements, [=](id<1> index) {
             const uint32_t encoded_index = index;

             // columns which are batch size
             const uint32_t i = encoded_index / loc_padded_output_width;
             const uint32_t j = encoded_index - i * loc_padded_output_width;

             const uint32_t idx = i * loc_padded_output_width + j;
             const uint32_t unpadded_idx = i * loc_input_stride + j;

             if (j >= loc_n_dims_to_encode)
                 loc_out[idx] = (T)1; //pad with 1
             else
                 loc_out[idx] = loc_in[unpadded_idx] * loc_scale + loc_offset;
         }).wait();
        return std::make_unique<Context>();
    }

    void backward_impl(const Context &ctx, const DeviceMatrixView<float> input,
                       const DeviceMatrixView<T> dL_doutput, DeviceMatrixView<T> *gradients = nullptr,
                       DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        throw std::logic_error("Not yet implemented.");

        if (!dL_dinput) return;

        const size_t n_elements = input.n() * this->get_input_width();
        if (n_elements <= 0) return; // nothing to do

        float *const dL_dx = dL_dinput->data();
        T const *const dL_dy = dL_doutput.GetPointer();
        auto loc_scale = m_scale;

        this->get_queue().parallel_for(n_elements, [=](auto item) {
            const size_t idx = item;
            dL_dx[idx] = (float)dL_dy[idx] * loc_scale;
        });
    }

  private:
    const float m_scale;
    const float m_offset;
};