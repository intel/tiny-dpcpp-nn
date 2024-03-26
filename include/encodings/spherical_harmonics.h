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
    SphericalHarmonicsEncoding(const uint32_t degree, const uint32_t n_dims_to_encode)
        : m_degree{degree}, m_n_dims_to_encode{n_dims_to_encode}, m_n_output_dims(degree * degree), m_n_to_pad(0) {

        if (n_dims_to_encode != 3) throw std::runtime_error{"Can only encode 3D directions in spherical harmonics."};

        if (m_degree <= 0) throw std::runtime_error{"Spherical harmonics must have positive degree."};

        if (m_degree > 8) throw std::runtime_error{"Spherical harmonics are only implemented up to degree 8."};
    }

    std::unique_ptr<Context> forward_impl(sycl::queue *const q, const DeviceMatrixView<float> input,
                                          DeviceMatrixView<T> *output = nullptr, bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {

        const uint32_t n_rows = input.m();
        if (!output || padded_output_width() == 0) return std::make_unique<Context>();
        if (input.n() != m_n_dims_to_encode)
            throw std::invalid_argument("input dimensions do not coincide with encoder");
        if (output->m() != input.m()) throw std::invalid_argument("Differing row numbers");
        if (output->n() != padded_output_width())
            throw std::invalid_argument("number of cols has to be padded output width.");

        // Wrap our data variable in a buffer
        float const *const loc_in = input.GetPointer();
        T *const loc_out = output->GetPointer();

        auto loc_stride = input.n();
        auto loc_degree = m_degree;
        auto loc_n_to_pad = m_n_to_pad;
        auto loc_padded_output_width = padded_output_width();
        auto loc_n_output_dims = m_n_output_dims;
        // Create a command group to issue commands to the queue
        q->parallel_for(range<1>(n_rows), [=](id<1> index) {
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

    void backward_impl(sycl::queue *const q, const Context &ctx, const DeviceMatrixView<float> input,
                       const DeviceMatrixView<T> dL_doutput, DeviceMatrixView<T> *gradients = nullptr,
                       DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {
        throw std::logic_error("Spherical backward not yet implemented.");
    }

    uint32_t input_width() const override { return m_n_dims_to_encode; }

    uint32_t padded_output_width() const override { return m_n_output_dims + m_n_to_pad; }

    uint32_t output_width() const override { return padded_output_width(); }

    void set_padded_output_width(uint32_t padded_output_width) override {
        if (padded_output_width < m_n_output_dims) {
            throw std::invalid_argument(
                "Padded width has to be larger than unpadded. m_n_output_dims: " + std::to_string(m_n_output_dims) +
                ", padded_output_width: " + std::to_string(padded_output_width));
        }

        m_n_to_pad = padded_output_width - m_n_output_dims;
    }

    void initialize_params(float *params_full_precision, float scale = 1) override {}

  private:
    uint32_t m_degree;
    uint32_t m_n_dims_to_encode;

    // derived sizes
    uint32_t m_n_output_dims;
    uint32_t m_n_to_pad;
};
