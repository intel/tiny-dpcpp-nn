/**
 * @file spherical_harmonics.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of spherical harmonics encoding class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// Encoding which takes exactly 3 input cols and a spherical harmonics
// degree and a padding. Performs spherical harmonics + identity encoding
template <typename T> class SphericalHarmonicsEncoding : public Encoding<T> {
  public:
    SphericalHarmonicsEncoding(const uint32_t degree, const uint32_t n_dims_to_encode, 
        const uint32_t padded_output_width, sycl::queue& Q)
        : Encoding<T>(n_dims_to_encode, degree*degree, padded_output_width, Q), m_degree{degree} {

        if (this->get_input_width() != 3) throw std::runtime_error{"Can only encode 3D directions in spherical harmonics."};

        if (m_degree == 0 || m_degree > 8) throw std::runtime_error{"Spherical harmonics degree > 0 and <= 8."};
    }

    std::unique_ptr<Context> forward_impl(const DeviceMatrixView<float> input,
                                          DeviceMatrixView<T> *output = nullptr, bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {

        const size_t n_rows = input.m();
        if (!output || this->get_padded_output_width() == 0) return std::make_unique<Context>();
        if (input.n() != this->get_input_width())
            throw std::invalid_argument("input dimensions do not coincide with encoder");
        if (use_inference_params) throw std::invalid_argument("Cannot yet use inference params");
        if (output->m() != input.m()) throw std::invalid_argument("Differing row numbers");
        if (output->n() != this->get_padded_output_width())
            throw std::invalid_argument("number of cols has to be padded output width.");

        // Wrap our data variable in a buffer
        float const *const loc_in = input.GetPointer();
        T *const loc_out = output->GetPointer();

        auto loc_stride = input.n();
        auto loc_degree = m_degree;
        auto loc_n_to_pad = this->get_n_to_pad();
        auto loc_padded_output_width = this->get_padded_output_width();
        auto loc_n_output_dims = this->get_output_width();
        
        this->get_queue().parallel_for(range<1>(n_rows), [=](id<1> index) {
            const uint32_t row = index;

            for (uint32_t j = 0; j < loc_n_to_pad; ++j) {
                loc_out[row * loc_padded_output_width + (loc_n_output_dims + j)] = (T)1.0f;
            }

            // this does degree^2 contiguous elements
            sh_enc<T>(loc_degree, loc_in[0 + row * loc_stride] * 2.f - 1.f, loc_in[1 + row * loc_stride] * 2.f - 1.f,
                      loc_in[2 + row * loc_stride] * 2.f - 1.f, loc_out, row * loc_padded_output_width);
        });

        return std::make_unique<Context>();
    }

    void backward_impl(const Context &ctx, const DeviceMatrixView<float> input,
                       const DeviceMatrixView<T> dL_doutput, DeviceMatrixView<T> *gradients = nullptr,
                       DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        throw std::logic_error("Spherical backward not yet implemented.");
    }

  private:
    const uint32_t m_degree;
};
