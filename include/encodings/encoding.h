/**
 * @file encoding.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of an absract base class for the encodings.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <common.h>
#include <type_traits> // for std::is_same

#include <cstdint>
#include <sycl/sycl.hpp>

#include "DeviceMatrix.h"
#include "DeviceMem.h"
#include "common.h"
#include "json.hpp"

using json = nlohmann::json;

struct EncodingParams {
    inline static const std::string ENCODING = "otype";                            // EncodingNames
    inline static const std::string N_DIMS_TO_ENCODE = "n_dims_to_encode";         // uint32_t
    inline static const std::string GRID_TYPE = "type";                            // GridType Hash, Dense, Tiled
    inline static const std::string N_LEVELS = "n_levels";                         // uint32_t
    inline static const std::string N_FEATURES = "n_features";                     // uint32_t
    inline static const std::string N_FEATURES_PER_LEVEL = "n_features_per_level"; // uint32_t
    inline static const std::string LOG2_HASHMAP_SIZE = "log2_hashmap_size";       // uint32_t
    inline static const std::string BASE_RESOLUTION = "base_resolution";           // uint32_t
    inline static const std::string PER_LEVEL_SCALE = "per_level_scale";           // float
    inline static const std::string DEGREE = "degree";                             // uint32_t
    inline static const std::string SCALE = "scale";                               // float
    inline static const std::string OFFSET = "offset";                             // float
    inline static const std::string HASH = "hash";                                 // HashType
    inline static const std::string INTERPOLATION_METHOD = "interpolation";        // InterpolationType
    inline static const std::string USE_STOCHASTIC_INTERPOLATION = "stochastic_interpolation"; // bool
};

struct EncodingNames {
    inline static const std::string IDENTITY = "Identity";
    inline static const std::string SPHERICALHARMONICS = "SphericalHarmonics";
    inline static const std::string GRID = "HashGrid";
};

enum class GradientMode {
    Ignore,
    Overwrite,
    Accumulate,
};

enum class GridType {
    Hash,
    Dense,
    Tiled,
};

enum class HashType {
    Prime,
    CoherentPrime,
    ReversedPrime,
    Rng,
};

enum class InterpolationType {
    Nearest,
    Linear,
    Smoothstep,
};

enum class ReductionType {
    Concatenation,
    Sum,
    Product,
};

template <typename T> class Encoding {
  public:
    Encoding() = delete;
    Encoding(const uint32_t input_width, const uint32_t output_width, const uint32_t padded_output_width, sycl::queue& Q) :
        m_params(nullptr), m_input_width(input_width), m_output_width(output_width), 
        m_padded_output_width(padded_output_width), m_q(Q) {
            if (m_input_width == 0) throw std::invalid_argument("Input width cannot be zero");
            if (m_output_width == 0) throw std::invalid_argument("Output width cannot be zero");
            if (m_padded_output_width < m_output_width) throw std::invalid_argument("Padded output width cannot be less than output width");
        }
    virtual ~Encoding() {}

    virtual std::unique_ptr<Context> forward_impl(const DeviceMatrixView<float> input,
                                                  DeviceMatrixView<T> *output = nullptr,
                                                  bool use_inference_params = false,
                                                  bool prepare_input_gradients = false) = 0;

    virtual void backward_impl(const Context &ctx, const DeviceMatrixView<float> input,
                               const DeviceMatrixView<T> dL_doutput, DeviceMatrixView<T> *gradients = nullptr,
                               DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                               GradientMode param_gradients_mode = GradientMode::Overwrite) = 0;

    // virtual void set_padded_output_width(uint32_t padded_output_width) = 0;

    
    // TODO: should be inherited from object.h at some point
    // Get the params of the encoding. Use this to initialize them to values.
    std::shared_ptr<DeviceMatrix<T> > get_params() const { return m_params; }

    //get the input width
    uint32_t get_input_width() const {return m_input_width;}

    //get the padded output width. This is in general a power of 2
    uint32_t get_padded_output_width() const {return m_padded_output_width;}

    ///get the not-padded output width. This may or may not be the same as the padded width
    uint32_t get_output_width() const {return m_output_width;}

    uint32_t get_n_to_pad() const {return m_padded_output_width - m_output_width;}

    size_t get_n_params() const { return m_params ? m_params->size() : 0; }

    sycl::queue& get_queue() { return m_q; }

    // ///Function which takes a pointer to a DeviceMatrix and assigns it to the m_params
    // ///Throws an error if the input pointer is a nullptr or if the sizes do not match with m_n_params
    // void set_params(DeviceMatrix<T> * const params_full_precision) {

    //     static_assert(std::is_same<T, float>::value, "Only float are supported");

    //     if (!params_full_precision) throw std::invalid_argument("Cannot set params with a nullptr");
    //     if (params_full_precision->size() != n_params()) throw std::invalid_argument("Parameter size mismatch");

    //     m_params = params_full_precision;
    // }

  protected:
    ///Parameters or Weights. They are owned by this encoding.
    std::shared_ptr<DeviceMatrix<T> > m_params;

    //Not used
    // struct ForwardContext : public Context {
    //     DeviceMatrix<T> network_input;
    //     std::unique_ptr<Context> encoding_ctx;
    //     std::unique_ptr<Context> network_ctx;
    // };

    const uint32_t m_input_width;
    const uint32_t m_output_width;
    const uint32_t m_padded_output_width;

    sycl::queue& m_q;
};
