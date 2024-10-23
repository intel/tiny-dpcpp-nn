/**
 * @file frequency.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of the frequency encoding class.
 * @version 0.1
 * @date 2024-10-23
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cmath>
#include <stdexcept>

#include "encoding.h"

template <typename T>
class FrequencyEncoding : public Encoding<T> {
 public:
  FrequencyEncoding(const uint32_t n_frequencies,
                    const uint32_t n_dims_to_encode,
                    const uint32_t padded_output_width, sycl::queue &Q)
      : Encoding<T>(n_dims_to_encode, n_dims_to_encode * n_frequencies * 2,
                    padded_output_width, Q),
        n_frequencies_{n_frequencies} {}

  std::unique_ptr<Context> forward_impl(
      const DeviceMatrixView<float> input,
      DeviceMatrixView<T> *output = nullptr, bool use_inference_params = false,
      bool prepare_input_gradients = false) override {
    if (!output) return std::make_unique<Context>();
    if (input.n() != this->get_input_width())
      throw std::invalid_argument(
          "input dimensions do not coincide with encoder");
    if (use_inference_params)
      throw std::invalid_argument("Cannot yet use inference params");
    if (output->m() != input.m())
      throw std::invalid_argument("Differing row numbers");
    if (output->n() != this->get_padded_output_width())
      throw std::invalid_argument(
          "number of cols has to be padded output width.");

    auto forward = std::make_unique<Context>();

    float *loc_dy_dx = nullptr;
    if (prepare_input_gradients) {
      forward = std::make_unique<ForwardContext>(
          input.m(), this->get_output_width(), this->get_queue());
      dynamic_cast<ForwardContext *>(forward.get())
          ->dy_dx.fill(0.0f);  // may not be necessary
      loc_dy_dx = dynamic_cast<ForwardContext *>(forward.get())->dy_dx.data();
    }

    {
      // copy the data to avoid implicit copy of 'this'
      const size_t num_elements = input.m() * this->get_padded_output_width();
      const uint32_t output_width = this->get_output_width();
      const uint32_t padded_output_width = this->get_padded_output_width();
      const uint32_t n_frequencies = n_frequencies_;

      // works always since we checked above, need to do this to copy the object
      auto loc_output = *output;

      this->get_queue()
          .parallel_for(
              num_elements,
              [=](sycl::id<1> index) {
                const size_t encoded_index = index;

                const uint32_t i = encoded_index / padded_output_width;
                const uint32_t j = encoded_index - i * padded_output_width;

                if (j >= output_width) {
                  loc_output(i, j) = (T)1;
                } else {
                  const uint32_t encoded_input_feature_j =
                      j / (n_frequencies * 2);
                  const uint32_t log2_frequency = (j / 2) % n_frequencies;

                  const float phase_shift = (j % 2) * (M_PI / 2.0f);

                  const float x = input(i, encoded_input_feature_j) *
                                      ((size_t)1 << log2_frequency) * M_PI +
                                  phase_shift;
                  loc_output(i, j) = (T)sycl::sin(x);
                  if (loc_dy_dx != nullptr) {
                    loc_dy_dx[i * output_width + j] =
                        (float)((size_t)1 << log2_frequency) * M_PI *
                        sycl::cos(x);
                  }
                }
              })
          .wait();
    }

    return forward;
  }

  void backward_impl(
      const Context &ctx, const DeviceMatrixView<float> input,
      const DeviceMatrixView<T> dL_doutput,
      DeviceMatrixView<T> *gradients = nullptr,
      DeviceMatrix<float> *dL_dinput = nullptr,
      bool use_inference_params = false,
      GradientMode param_gradients_mode = GradientMode::Overwrite) override {
    if (input.m() == 0) throw std::invalid_argument("batch_size == 0");
    if (!dL_dinput) return;

    if (use_inference_params)
      throw std::invalid_argument("Cannot use inference params.");

    {
      // copy to local to avoid implciit copy of 'this'
      const uint32_t n_dims_to_encode = this->get_input_width();
      const size_t num_elements = input.m() * n_dims_to_encode;
      const uint32_t n_frequencies = n_frequencies_;
      float const *const dy_dx =
          dynamic_cast<const ForwardContext &>(ctx).dy_dx.data();
      auto loc_dL_dinput = dL_dinput->GetView();
      this->get_queue()
          .parallel_for(
              num_elements,
              [=](sycl::id<1> index) {
                const uint32_t encoded_index = index;

                const uint32_t i = encoded_index / n_dims_to_encode;
                const uint32_t j = encoded_index - i * n_dims_to_encode;

                const uint32_t outputs_per_input = n_frequencies * 2;

                float result = 0.0f;
                for (int k = 0; k < outputs_per_input; ++k) {
                  result += (float)dL_doutput(i, j * outputs_per_input + k) *
                            dy_dx[i * n_dims_to_encode * outputs_per_input +
                                  j * outputs_per_input + k];
                }
                loc_dL_dinput(i, j) = result;
              })
          .wait();
    }
  }

 private:
  struct ForwardContext : public Context {
    ForwardContext(const size_t n_rows, const size_t n_cols, sycl::queue &Q)
        : dy_dx(n_rows, n_cols, Q) {}

    DeviceMatrix<float> dy_dx;
  };

  const uint32_t n_frequencies_;
};