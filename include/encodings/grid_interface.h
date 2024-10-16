/**
 * @file grid_interface.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of an abstract grid encoding class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <common.h>
#include <encoding.h>

#include <cstdint>
#include <sycl/sycl.hpp>

namespace tinydpcppnn {
namespace encodings {
namespace grid {

static constexpr uint32_t MAX_N_LEVELS = 128;
struct GridOffsetTable {
    uint32_t data[MAX_N_LEVELS + 1] = {};
    uint32_t size = 0;
};

template <typename T> class GridEncoding : public Encoding<T> {
  public:
    GridEncoding(const uint32_t input_width, const uint32_t output_width, 
        const uint32_t padded_output_width, sycl::queue& Q) : 
        Encoding<T>(input_width, output_width, padded_output_width, Q) {}
    GridEncoding() = delete;
    virtual ~GridEncoding() {}

    virtual uint32_t n_pos_dims() const = 0;
    virtual uint32_t n_features_per_level() const = 0;

    virtual size_t level_n_params(uint32_t level) const = 0;
    virtual size_t level_params_offset(uint32_t level) const = 0;

    virtual const GridOffsetTable &grid_offset_table() const = 0;

    float max_level() const { return m_max_level; }

    void set_max_level(float value) { m_max_level = value; }

  protected:
    // Disables lookups of finer levels than this.
    // The default value of 1000 effectively disables the feature
    float m_max_level = 1000.f;
};

} // namespace grid
} // namespace encodings
} // namespace tinydpcppnn