/**
 * @file grid.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Derived, templated grid encoding class with creation functions for all grid parameters.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "DeviceMem.h"
#include "common.h"
#include "common_device.h"
#include "encoding.h"
#include "grid_interface.h"
#include "vec.h"

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using json = nlohmann::json;

namespace tinydpcppnn {
namespace encodings {
namespace grid {
namespace kernels {

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
void kernel_grid(const uint32_t num_elements, const uint32_t num_grid_features, const GridOffsetTable &offset_table,
                 const uint32_t base_resolution, const float log2_per_level_scale, float max_level,
                 const InterpolationType interpolation_type, const GridType grid_type, T const *__restrict__ grid,
                 const DeviceMatrixView<float> &positions_in, DeviceMatrixView<T> encoded_positions,
                 const sycl::nd_item<3> &item) {
    assert(grid != nullptr && "grid is nullptr, expected a valid pointer");

    const uint32_t i = item.get_global_id(2);
    if (i >= num_elements) return;

    const uint32_t level = item.get_group(1); // <- the level is the same for all threads

    max_level = ((max_level * num_grid_features) / N_FEATURES_PER_LEVEL);
    if (level >= max_level + 1e-3f) {
        for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
            encoded_positions(i, level * N_FEATURES_PER_LEVEL + f) = (T)0;
        }

        return;
    }

    grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];
    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    float pos[N_POS_DIMS];
    float pos_derivative[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            pos_derivative[dim] = identity_derivative(pos[dim]);
            pos[dim] = identity_fun(pos[dim]);
        }
    } else {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            pos_derivative[dim] = smoothstep_derivative(pos[dim]);
            pos[dim] = smoothstep(pos[dim]);
        }
    }

    auto grid_val = [&](const tnn::uvec<N_POS_DIMS> &local_pos) {
        const uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL;
        return *(tnn::tvec<T, N_FEATURES_PER_LEVEL, sizeof(T) * N_FEATURES_PER_LEVEL> *)&grid[index];
    };

    if (interpolation_type == InterpolationType::Nearest) {
        auto result = grid_val(pos_grid);
        for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
            encoded_positions(i, level * N_FEATURES_PER_LEVEL + f) = result[f];
        }
    } else {
        // N-linear interpolation
        tnn::tvec<T, N_FEATURES_PER_LEVEL, sizeof(T) * N_FEATURES_PER_LEVEL> result((T)0);

        for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
            float weight = 1.0f;
            tnn::uvec<N_POS_DIMS> pos_grid_local;

            for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
                if ((idx & (1 << dim)) == 0) {
                    weight *= 1 - pos[dim];
                    pos_grid_local[dim] = pos_grid[dim];
                } else {
                    weight *= pos[dim];
                    pos_grid_local[dim] = pos_grid[dim] + 1;
                }
            }

            result = tnn::fma((T)weight, grid_val(pos_grid_local), result);
        }

        for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
            encoded_positions(i, level * N_FEATURES_PER_LEVEL + f) = result[f];
        }
    }
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL,
          uint32_t N_FEATURES_PER_THREAD, HashType HASH_TYPE>
void kernel_grid_backward(const size_t num_elements, const uint32_t num_grid_features,
                          const GridOffsetTable &offset_table, const uint32_t base_resolution,
                          const float log2_per_level_scale, float max_level, const bool stochastic_interpolation,
                          const InterpolationType interpolation_type, const GridType grid_type,
                          GRAD_T *__restrict__ grid_gradient, DeviceMatrixView<float> positions_in,
                          DeviceMatrixView<T> dL_dy, const sycl::nd_item<3> &item) {
    const size_t i = (item.get_global_id(2) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
    if (i >= num_elements) return;

    const uint32_t level = item.get_group(1); // <- the level is the same for all threads.
    const uint32_t feature = item.get_global_id(2) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

    max_level = ((max_level * num_grid_features) / N_FEATURES_PER_LEVEL);

    if (level >= max_level + 1e-3f) return;

    grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];
    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    auto add_grid_gradient = [&](const tnn::uvec<N_POS_DIMS> &local_pos,
                                 const tnn::tvec<GRAD_T, N_FEATURES_PER_THREAD> &grad, const float weight) {
        const uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL +
            feature;

        for (size_t i = 0; i < N_FEATURES_PER_THREAD; ++i) {
            sycl::atomic_ref<GRAD_T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_op(grid_gradient[index + i]);
            atomic_op += (GRAD_T)weight * grad[i];
        }
    };

    float pos[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            // pos[dim] = identity_fun(pos[dim]);
        }
    } else {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            pos[dim] = smoothstep(pos[dim]);
        }
    }

    tnn::tvec<T, N_FEATURES_PER_THREAD> grad;

    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
        grad[f] = dL_dy(i, level * N_FEATURES_PER_LEVEL + feature + f);
    }

    if (interpolation_type == InterpolationType::Nearest) {
        add_grid_gradient(pos_grid, grad, 1.0f);
    } else if (stochastic_interpolation) {
        const float sample = random_val(1337, i + level * num_elements);
        tnn::uvec<N_POS_DIMS> pos_grid_local;

        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos_grid_local[dim] = (sample >= pos[dim]) ? (pos_grid[dim]) : (pos_grid[dim] + 1);
        }

        add_grid_gradient(pos_grid_local, grad, 1.0f);
    } else { // N-linear interpolation

        for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
            float weight = 1.0f;
            tnn::uvec<N_POS_DIMS> pos_grid_local;

            for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
                if ((idx & (1 << dim)) == 0) {
                    weight *= 1 - pos[dim];
                    pos_grid_local[dim] = pos_grid[dim];
                } else {
                    weight *= pos[dim];
                    pos_grid_local[dim] = pos_grid[dim] + 1;
                }
            }

            add_grid_gradient(pos_grid_local, grad, weight);
        }
    }
}

template <typename T, uint32_t N_POS_DIMS>
void kernel_grid_backward_input(const size_t num_elements, const uint32_t num_grid_features,
                                DeviceMatrixView<T> dL_dy_rm, DeviceMatrixView<float> dy_dx,
                                DeviceMatrixView<float> dL_dx, const sycl::nd_item<1> &item) {
    const size_t i = item.get_global_linear_id();
    if (i >= num_elements) return;

    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
        dL_dx(i, dim) = 0.0f;
    }

    for (int k = 0; k < num_grid_features; ++k) {
        const float dL_dy_local = (float)dL_dy_rm(i, k);

        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            dL_dx(i, dim) += dL_dy_local * dy_dx(i + dim, k);
        }
    }
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL,
          uint32_t N_FEATURES_PER_THREAD, HashType HASH_TYPE>
void kernel_grid_backward_input_backward_grid(const size_t num_elements, const uint32_t num_grid_features,
                                              const GridOffsetTable &offset_table, const uint32_t base_resolution,
                                              const float log2_per_level_scale, float max_level,
                                              const InterpolationType interpolation_type, const GridType grid_type,
                                              const DeviceMatrixView<float> dL_ddLdx,
                                              const DeviceMatrixView<float> positions_in,
                                              const DeviceMatrixView<T> dL_dy, GRAD_T *__restrict__ grid_gradient,
                                              const sycl::nd_item<3> &item) {
    const size_t i = (item.get_global_id(2) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
    if (i >= num_elements) return;

    const uint32_t level = item.get_group(1);
    const uint32_t feature = item.get_global_id(2) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

    max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

    if (level >= max_level + 1e-3f) return;

    grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    auto add_grid_gradient = [&](const tnn::uvec<N_POS_DIMS> &local_pos,
                                 const tnn::tvec<GRAD_T, N_FEATURES_PER_THREAD> &grad, const float weight) {
        const uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL +
            feature;

        for (size_t i = 0; i < N_FEATURES_PER_THREAD; ++i) {
            sycl::atomic_ref<GRAD_T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic(grid_gradient +
                                                                                                     index + i);
            atomic.fetch_add(((GRAD_T)weight) * grad[i]);
        }
    };

    float pos[N_POS_DIMS];
    float pos_derivative[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            pos_derivative[dim] = identity_derivative(pos[dim]);
            pos[dim] = identity_fun(pos[dim]);
        }
    } else {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            pos_derivative[dim] = smoothstep_derivative(pos[dim]);
            pos[dim] = smoothstep(pos[dim]);
        }
    }

    tnn::tvec<T, N_FEATURES_PER_THREAD> grad;

    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
        grad[f] = dL_dy(i, level * N_FEATURES_PER_LEVEL + feature + f);
    }

    // d(dydx)_dgrid is zero when there's no interpolation.
    if (interpolation_type != InterpolationType::Nearest) {
        // for N-linear interpolation
        for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
            const float grad_in = scale * dL_ddLdx(i, grad_dim) * pos_derivative[grad_dim];

            for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
                float weight = grad_in;
                tnn::uvec<N_POS_DIMS> pos_grid_local;

                for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
                    const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;

                    if ((idx & 1 << non_grad_dim) == 0) {
                        weight *= 1 - pos[dim];
                        pos_grid_local[dim] = pos_grid[dim];
                    } else {
                        weight *= pos[dim];
                        pos_grid_local[dim] = pos_grid[dim] + 1;
                    }
                }

                // left
                pos_grid_local[grad_dim] = pos_grid[grad_dim];
                add_grid_gradient(pos_grid_local, grad, -weight);
                // right
                pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
                add_grid_gradient(pos_grid_local, grad, weight);
            }
        }
    }
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD,
          HashType HASH_TYPE>
void kernel_grid_backward_input_backward_input(const size_t num_elements, const uint32_t num_grid_features,
                                               const GridOffsetTable &offset_table, const uint32_t base_resolution,
                                               const float log2_per_level_scale, float max_level,
                                               const InterpolationType interpolation_type, const GridType grid_type,
                                               const DeviceMatrixView<float> dL_ddLdx,
                                               const DeviceMatrixView<float> positions_in,
                                               const DeviceMatrixView<T> dL_dy, const T *__restrict__ grid,
                                               DeviceMatrixView<float> dL_dx, const sycl::nd_item<3> &item) {
    const size_t i = (item.get_global_id(2) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
    if (i >= num_elements) return;

    const uint32_t level = item.get_group(1);
    const uint32_t feature = item.get_global_id(2) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

    max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

    if (level >= max_level + 1e-3f) return;

    grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
    const uint32_t hashmap_size = offset_table.data[level + 1] - offset_table.data[level];

    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint32_t resolution = grid_resolution(scale);

    float pos[N_POS_DIMS];
    float pos_derivative[N_POS_DIMS];
    float pos_2nd_derivative[N_POS_DIMS];
    tnn::uvec<N_POS_DIMS> pos_grid;

    if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            pos_2nd_derivative[dim] = identity_2nd_derivative(pos[dim]);
            pos_derivative[dim] = identity_derivative(pos[dim]);
            pos[dim] = identity_fun(pos[dim]);
        }
    } else {
        for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
            pos[dim] = sycl::fma(scale, positions_in(i, dim), 0.5f);
            const float tmp = sycl::floor(pos[dim]);
            pos_grid[dim] = (uint32_t)(int)tmp;
            pos[dim] -= tmp;
            pos_2nd_derivative[dim] = smoothstep_2nd_derivative(pos[dim]);
            pos_derivative[dim] = smoothstep_derivative(pos[dim]);
            pos[dim] = smoothstep(pos[dim]);
        }
    }

    tnn::tvec<T, N_FEATURES_PER_THREAD> grad;

    for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
        grad[f] = dL_dy(i, level * N_FEATURES_PER_LEVEL + feature + f);
    }

    // d(dydx)_dx is zero when there's no interpolation
    if (interpolation_type != InterpolationType::Nearest) return;

    // for N-linear interpolation
    auto calc_dLdx = [&](const tnn::uvec<N_POS_DIMS> &local_pos, const float weight) {
        const uint32_t index =
            grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL +
            feature;
        float dL_dx_dim = 0.0f;

        for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
            dL_dx_dim += (float)grid[index + f] * (float)grad[f] * weight;
        }

        return dL_dx_dim;
    };

    tnn::tvec<float, N_POS_DIMS> grad_in_diag;
    tnn::tvec<float, N_POS_DIMS> grad_in_other;

    for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
        // from diagonal part of Hessian
        grad_in_diag[grad_dim] = scale * scale * dL_ddLdx(i, grad_dim) * pos_2nd_derivative[grad_dim];
        // from other part of Hessian
        grad_in_other[grad_dim] =
            scale * scale * dL_ddLdx(i, grad_dim) * pos_derivative[grad_dim]; // will do " *
                                                                              // pos_derivative[real_other_grad_dim]
                                                                              // " later
    }

    for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
        float grad_out = 0.0f;

        for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS - 1)); ++idx) {
            // from diagonal part of Hessian; d(doutput_d[grad_dim])_d[grad_dim]
            // NOTE: LinearInterpolations' diagonal part is 0.
            if (interpolation_type == InterpolationType::Smoothstep) {
                float weight_2nd_diag = grad_in_diag[grad_dim];
                tnn::uvec<N_POS_DIMS> pos_grid_local;

                for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
                    const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim + 1) : non_grad_dim;
                    // real non_grad_dim
                    if ((idx & 1 << non_grad_dim) == 0) {
                        weight_2nd_diag *= 1 - pos[dim];
                        pos_grid_local[dim] = pos_grid[dim];
                    } else {
                        weight_2nd_diag *= pos[dim];
                        pos_grid_local[dim] = pos_grid[dim] + 1;
                    }
                }

                // left
                pos_grid_local[grad_dim] = pos_grid[grad_dim];
                grad_out += calc_dLdx(pos_grid_local, -weight_2nd_diag);
                // right
                pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
                grad_out += calc_dLdx(pos_grid_local, weight_2nd_diag);
            }

            // from other part of Hessian;
            // d(doutput_d[real_other_grad_dim])_d[grad_dim]
            if constexpr (N_POS_DIMS > 1) {
                for (uint32_t other_grad_dim = 0; other_grad_dim < N_POS_DIMS - 1; ++other_grad_dim) {
                    const uint32_t real_other_grad_dim =
                        other_grad_dim >= grad_dim ? (other_grad_dim + 1) : other_grad_dim;
                    float weight_2nd_other = grad_in_other[real_other_grad_dim] * pos_derivative[grad_dim];
                    tnn::uvec<N_POS_DIMS> pos_grid_local;

                    for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS - 1; ++non_grad_dim) {
                        // real non_grad_dim
                        const uint32_t dim = non_grad_dim >= real_other_grad_dim ? (non_grad_dim + 1) : non_grad_dim;
                        if ((idx & 1 << non_grad_dim) == 0) {
                            if (dim != grad_dim)
                                weight_2nd_other *= 1 - pos[dim];
                            else
                                weight_2nd_other *= -1;
                            pos_grid_local[dim] = pos_grid[dim];
                        } else {
                            if (dim != grad_dim) weight_2nd_other *= pos[dim];
                            pos_grid_local[dim] = pos_grid[dim] + 1;
                        }
                    }

                    // left
                    pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim];
                    grad_out += calc_dLdx(pos_grid_local, -weight_2nd_other);
                    // right
                    pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim] + 1;
                    grad_out += calc_dLdx(pos_grid_local, weight_2nd_other);
                }
            }
        }

        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_target(
            dL_dx(i, grad_dim));
        atomic_target.fetch_add(grad_out);
    }
}

template <typename T, uint32_t N_POS_DIMS>
void kernel_grid_backward_input_backward_dLdoutput(const size_t num_elements, const uint32_t num_grid_features,
                                                   const DeviceMatrixView<float> dL_ddLdx,
                                                   const DeviceMatrixView<float> dy_dx, DeviceMatrixView<T> dL_ddLdy,
                                                   const sycl::nd_item<1> &item) {
    const size_t i = item.get_global_linear_id();
    if (i >= num_elements) return;

    for (uint32_t k = 0; k < num_grid_features; ++k) {
        dL_ddLdy(i, k) = (T)0;

        for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
            dL_ddLdy(i, k) += dy_dx(i + grad_dim, k) * dL_ddLdx(i, grad_dim);
        }
    }
}
} // namespace kernels

template <typename T, uint32_t N_POS_DIMS = 3, uint32_t N_FEATURES_PER_LEVEL = 2,
          HashType HASH_TYPE = HashType::CoherentPrime>
class GridEncodingTemplated : public GridEncoding<T> {
  public:
    using grad_t = float;

    GridEncodingTemplated(const uint32_t padded_output_width, const uint32_t n_features, 
        const uint32_t log2_hashmap_size, const uint32_t base_resolution,
        const float per_level_scale, const bool stochastic_interpolation, const InterpolationType interpolation_type,
        const GridType grid_type, sycl::queue& Q) : 
          GridEncoding<T>(N_POS_DIMS, n_features, padded_output_width, Q), 
          m_n_features{n_features}, 
          m_log2_hashmap_size{log2_hashmap_size}, m_base_resolution{base_resolution},
          m_per_level_scale{per_level_scale}, m_stochastic_interpolation{stochastic_interpolation},
          m_interpolation_type{interpolation_type}, m_grid_type{grid_type} 
    {

        m_n_levels = tinydpcppnn::math::div_round_up(m_n_features, N_FEATURES_PER_LEVEL);
        uint32_t offset = 0;

        if (m_n_levels > MAX_N_LEVELS)
            throw std::runtime_error("GridEncoding: m_n_levels= " + std::to_string(m_n_levels) + 
                " must be at most MAX_N_LEVELS= " + std::to_string(MAX_N_LEVELS));

        for (uint32_t i = 0; i < m_n_levels; ++i) {
            // Compute number of dense params required for the given level
            const uint32_t resolution = grid_resolution(grid_scale(i, std::log2(per_level_scale), base_resolution));

            uint32_t max_params = std::numeric_limits<uint32_t>::max() / 2;
            uint32_t params_in_level = std::pow((float)resolution, N_POS_DIMS) > (float)max_params
                                           ? max_params
                                           : tinydpcppnn::math::powi(resolution, N_POS_DIMS);

            // Make sure memory accesses will be aligned
            params_in_level = tinydpcppnn::math::next_multiple(params_in_level, 8u);

            if (m_grid_type == GridType::Dense) {
            } // No-op
            else if (m_grid_type == GridType::Tiled)
                params_in_level = std::min(params_in_level, tinydpcppnn::math::powi(base_resolution, N_POS_DIMS));
            else if (m_grid_type == GridType::Hash)
                params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));
            else
                throw std::invalid_argument("GridEncoding: invalid grid type " + std::to_string((int)m_grid_type));

            m_offset_table.data[i] = offset;
            offset += params_in_level;
        }

        m_offset_table.data[m_n_levels] = offset;
        m_offset_table.size = m_n_levels + 1;

        // allocate the params
        this->m_params = std::make_shared<DeviceMatrix<T>>(m_offset_table.data[m_n_levels], N_FEATURES_PER_LEVEL, this->get_queue());
        this->m_params->fill_random(-1e-4f, 1e-4f).wait(); //init params 
        // this->m_n_params = m_offset_table.data[m_n_levels] * N_FEATURES_PER_LEVEL;

        if (n_features % N_FEATURES_PER_LEVEL != 0)
            throw std::runtime_error("GridEncoding: n_features= " + std::to_string(n_features) + " must be a multiple of N_FEATURES_PER_LEVEL = {} " +
                                    std::to_string(N_FEATURES_PER_LEVEL));
    }

    std::unique_ptr<Context> forward_impl(const DeviceMatrixView<float> input,
                                          DeviceMatrixView<T> *output = nullptr, bool use_inference_params = false,
                                          bool prepare_input_gradients = false) override {

        if (this->get_padded_output_width() == 0) throw std::invalid_argument("Can't have width == 0");
        if (use_inference_params) throw std::invalid_argument("Can't use inference params.");
        if (prepare_input_gradients) throw std::invalid_argument("Can't prepare input gradients");
        if (!output) return nullptr;
        if (output->n() != this->get_padded_output_width())
            throw std::invalid_argument("Dimension mismatch grid encoding forw_impl output");
        if (input.n() != this->get_input_width())
            throw std::invalid_argument("Dimension mismatch grid encoding forw_impl input");

        const uint32_t batch_size = input.m();

        // zero the padded values, i.e., the last m_n_to_pad values of each input
        {
            auto out = output->GetPointer() + this->get_output_width();
            const size_t bytes_to_zero = this->get_n_to_pad() * sizeof(T);
            const uint32_t stride = this->get_padded_output_width();

            for (int iter = 0; iter < batch_size; iter++) {
                this->get_queue().memset(out + iter * stride, 0, bytes_to_zero);
            }

            this->get_queue().wait();
        }

        // Idea: each block only takes care of _one_ hash level (but may iterate
        // over multiple input elements). This way, only one level of the hashmap
        // needs to fit into caches at a time (and it reused for consecutive
        // elements) until it is time to process the next level.

        {
            static constexpr uint32_t N_THREADS_HASHGRID = 512;
            const sycl::range<3> blocks_hashgrid(1, m_n_levels,
                                                 tinydpcppnn::math::div_round_up(batch_size, N_THREADS_HASHGRID));
            const uint32_t loc_n_features = m_n_features;
            const auto loc_offset_table = m_offset_table;
            const uint32_t loc_base_resolution = m_base_resolution;
            const float loc_log2_per_level_scale = std::log2(m_per_level_scale);
            const float loc_max_level = GridEncoding<T>::m_max_level;
            const auto loc_interpolation_type = m_interpolation_type;
            const auto loc_grid_type = m_grid_type;
            T const *loc_weights = this->get_params()->data();
            const DeviceMatrixView<float> loc_input_view = input;
            DeviceMatrixView<T> loc_output_view = *output;
            this->get_queue().parallel_for(sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREADS_HASHGRID),
                                              sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                            [=](sycl::nd_item<3> item) {
                                kernels::kernel_grid<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>(
                                    batch_size, loc_n_features, loc_offset_table, loc_base_resolution,
                                    loc_log2_per_level_scale, loc_max_level, loc_interpolation_type, loc_grid_type,
                                    loc_weights, loc_input_view, loc_output_view, item);
                            })
                .wait();
        }

        return nullptr;
    }

    void backward_impl(const Context &ctx, const DeviceMatrixView<float> input,
                       const DeviceMatrixView<T> dL_doutput, DeviceMatrixView<T> *gradients,
                       DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                       GradientMode param_gradients_mode = GradientMode::Overwrite) override {

        const size_t batch_size = input.m();
        if (batch_size == 0) throw std::invalid_argument("batch_size == 0");
        if (use_inference_params) throw std::invalid_argument("Cannot use inference params.");
        static_assert(std::is_same<grad_t, T>::value);

        static constexpr size_t N_THREADS_HASHGRID = 256;
        static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

        if (param_gradients_mode != GradientMode::Ignore) {
            // currently we can only do grad_t == T.
            // Later we need to allocate a DeviceMem for intermediate gradients
            // of type grad_t
            grad_t *const grid_gradient = gradients->GetPointer();

            if (param_gradients_mode == GradientMode::Overwrite) {
                this->get_queue().memset(grid_gradient, 0, this->get_n_params() * sizeof(grad_t)).wait();
            }

            this->get_queue().submit([&](sycl::handler &cgh) {
                auto loc_n_features = m_n_features;
                auto loc_offset_table = m_offset_table;
                auto loc_base_resolution = m_base_resolution;
                auto loc_log2_per_level_scale = std::log2(m_per_level_scale);
                auto loc_max_level = this->m_max_level;
                auto loc_stochastic_interpolation = m_stochastic_interpolation;
                auto loc_interpolation_type = m_interpolation_type;
                auto loc_grid_type = m_grid_type;
                DeviceMatrixView<float> loc_input = input;
                DeviceMatrixView<T> loc_output = dL_doutput;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, m_n_levels,
                                                     tinydpcppnn::math::next_multiple(
                                                         batch_size * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD,
                                                         N_THREADS_HASHGRID)),
                                      sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                    [=](sycl::nd_item<3> item) {
                        kernels::kernel_grid_backward<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL,
                                                      N_FEATURES_PER_THREAD, HASH_TYPE>(
                            batch_size, loc_n_features, loc_offset_table, loc_base_resolution, loc_log2_per_level_scale,
                            loc_max_level, loc_stochastic_interpolation, loc_interpolation_type, loc_grid_type,
                            grid_gradient, loc_input, loc_output, item);
                    });
            });

            // TODO: if we would use intermediate array for different grid gradients,
            // copy back to original at this point
        }

        if (dL_dinput) {
            // remove
            const auto &forward = dynamic_cast<const ForwardContext &>(ctx);
            auto loc_n_features = m_n_features;
            DeviceMatrixView<T> loc_output = dL_doutput;
            DeviceMatrixView<float> loc_dy_dx = forward.dy_dx.GetView();
            DeviceMatrixView<float> loc_input = dL_dinput->GetView();
            this->get_queue().parallel_for(
                sycl::nd_range<1>(tinydpcppnn::math::next_multiple(batch_size, N_THREADS_HASHGRID), N_THREADS_HASHGRID),
                [=](sycl::nd_item<1> item) {
                    kernels::kernel_grid_backward_input<T, N_POS_DIMS>(batch_size, loc_n_features, loc_output,
                                                                       loc_dy_dx, loc_input, item);
                });
        }
    }

    void backward_backward_input_impl(const Context &ctx, const DeviceMatrix<float> &input,
                                      const DeviceMatrix<float> &dL_ddLdinput, const DeviceMatrix<T> &dL_doutput,
                                      DeviceMatrix<T> *dL_ddLdoutput = nullptr,
                                      DeviceMatrix<float> *dL_dinput = nullptr, bool use_inference_params = false,
                                      GradientMode param_gradients_mode = GradientMode::Overwrite) {
        const size_t batch_size = input.m();
        if (batch_size == 0) throw std::invalid_argument("batch_size == 0");
        if (this->padded_output_width() == 0) throw std::invalid_argument("Padded output width == 0");
        if (use_inference_params) throw std::invalid_argument("Cannot use inference params.");
        static_assert(std::is_same<grad_t, T>::value);

        static constexpr size_t N_THREADS_HASHGRID = 256;
        static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

        const auto &forward = dynamic_cast<const ForwardContext &>(ctx);

        if (param_gradients_mode != GradientMode::Ignore) {
            grad_t *const grid_gradient = this->gradients();

            if (param_gradients_mode == GradientMode::Overwrite) {
                this->get_queue().memset(grid_gradient, 0, this->n_params() * sizeof(grad_t)).wait();
            }

            // from dL_d(dL_dx) to dL_dgrid
            this->get_queue().submit([&](sycl::handler &cgh) {
                auto loc_n_features = m_n_features;
                auto loc_offset_table = m_offset_table;
                auto loc_base_resolution = m_base_resolution;
                auto loc_log2_per_level_scale = std::log2(m_per_level_scale);
                auto loc_max_level = this->m_max_level;
                auto loc_interpolation_type = m_interpolation_type;
                auto loc_grid_type = m_grid_type;
                DeviceMatrixView<float> loc_dL_ddLdinput = dL_ddLdinput.GetView();
                DeviceMatrixView<float> loc_input = input.GetView();
                DeviceMatrixView<T> loc_output = dL_doutput.GetView();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, m_n_levels,
                                                     tinydpcppnn::math::next_multiple(
                                                         batch_size * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD,
                                                         N_THREADS_HASHGRID)),
                                      sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                    [=](sycl::nd_item<3> item) {
                        kernels::kernel_grid_backward_input_backward_grid<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL,
                                                                          N_FEATURES_PER_THREAD, HASH_TYPE>(
                            batch_size, loc_n_features, loc_offset_table, loc_base_resolution, loc_log2_per_level_scale,
                            loc_max_level, loc_interpolation_type, loc_grid_type, loc_dL_ddLdinput, loc_input,
                            loc_output, grid_gradient, item);
                    });
            });
        }

        if (dL_ddLdoutput) {
            auto loc_n_features = m_n_features;
            DeviceMatrixView<float> loc_input = dL_ddLdinput.GetView();
            DeviceMatrixView<float> loc_dy_dx = forward.dy_dx.GetView();
            DeviceMatrixView<T> loc_output = dL_ddLdoutput->GetView();
            this->get_queue().parallel_for(
                sycl::nd_range<1>(tinydpcppnn::math::next_multiple(batch_size, N_THREADS_HASHGRID), N_THREADS_HASHGRID),
                [=](sycl::nd_item<1> item) {
                    kernels::kernel_grid_backward_input_backward_dLdoutput<T, N_POS_DIMS>(
                        batch_size, loc_n_features, loc_input, loc_dy_dx, loc_output, item);
                });
        }

        if (dL_dinput) {
            // from dL_d(dL_dx) to dL_dx
            this->get_queue().submit([&](sycl::handler &cgh) {
                auto loc_n_features = m_n_features;
                auto loc_offset_table = m_offset_table;
                auto loc_base_resolution = m_base_resolution;
                auto loc_log2_per_level_scale = std::log2(m_per_level_scale);
                auto loc_max_level = this->m_max_level;
                auto loc_interpolation_type = m_interpolation_type;
                auto loc_grid_type = m_grid_type;
                DeviceMatrixView<float> loc_ddLdinput = dL_ddLdinput.GetView();
                DeviceMatrixView<float> loc_input = input.GetView();
                DeviceMatrixView<T> loc_doutput = dL_doutput.GetView();
                T *const weights = this->params();
                DeviceMatrixView<float> loc_dinput = dL_dinput->GetView();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, m_n_levels,
                                                     tinydpcppnn::math::next_multiple(
                                                         batch_size * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD,
                                                         N_THREADS_HASHGRID)),
                                      sycl::range<3>(1, 1, N_THREADS_HASHGRID)),
                    [=](sycl::nd_item<3> item) {
                        kernels::kernel_grid_backward_input_backward_input<T, N_POS_DIMS, N_FEATURES_PER_LEVEL,
                                                                           N_FEATURES_PER_THREAD, HASH_TYPE>(
                            batch_size, loc_n_features, loc_offset_table, loc_base_resolution, loc_log2_per_level_scale,
                            loc_max_level, loc_interpolation_type, loc_grid_type, loc_ddLdinput, loc_input, loc_doutput,
                            weights, loc_dinput, item);
                    });
            });
        }
    }

    size_t level_n_params(uint32_t level) const override {
        return level_params_offset(level + 1) - level_params_offset(level);
    }

    size_t level_params_offset(uint32_t level) const override {
        if (level >= m_offset_table.size) throw std::runtime_error{"Out of bounds params offset request."};
        return m_offset_table.data[level];
    }

    const GridOffsetTable &grid_offset_table() const override { return m_offset_table; }

    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const // TODO: override
    {
        // Even though we have parameters, they can't really be considered a
        // "layer". So we return an empty array here.
        return {};
    }

    uint32_t n_pos_dims() const override { return N_POS_DIMS; }

    uint32_t n_features_per_level() const override { return N_FEATURES_PER_LEVEL; }

    json hyperparams() const // TODO: override
    {
        json result = {
            {EncodingParams::ENCODING, EncodingNames::GRID},
            {EncodingParams::GRID_TYPE, m_grid_type},
            {EncodingParams::N_LEVELS, m_n_levels},
            {EncodingParams::N_FEATURES_PER_LEVEL, N_FEATURES_PER_LEVEL},
            {EncodingParams::BASE_RESOLUTION, m_base_resolution},
            {EncodingParams::PER_LEVEL_SCALE, m_per_level_scale},
            {EncodingParams::INTERPOLATION_METHOD, m_interpolation_type},
            {EncodingParams::HASH, HASH_TYPE},
        };

        if (m_grid_type == GridType::Hash) {
            result[EncodingParams::LOG2_HASHMAP_SIZE] = m_log2_hashmap_size;
        }

        return result;
    }

  private:
    struct ForwardContext : public Context {
        DeviceMatrix<float, MatrixLayout::RowMajor> positions;
        DeviceMatrix<float, MatrixLayout::RowMajor> dy_dx;
    };

    uint32_t m_n_features;
    uint32_t m_n_levels;
    GridOffsetTable m_offset_table;
    uint32_t m_log2_hashmap_size;
    uint32_t m_base_resolution;

    uint32_t m_n_dims_to_pass_through;

    float m_per_level_scale;

    // uint32_t this->m_n_params;
    bool m_stochastic_interpolation;
    InterpolationType m_interpolation_type;
    GridType m_grid_type;
};

template <typename T, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
std::shared_ptr<GridEncoding<T>> create_grid_encoding_templated_2(const json &encoding, 
    std::optional<uint32_t> padded_output_width, sycl::queue& Q) 
{

    const uint32_t log2_hashmap_size = encoding.value(EncodingParams::LOG2_HASHMAP_SIZE, 19u);
    const uint32_t n_dims_to_encode = encoding.value(EncodingParams::N_DIMS_TO_ENCODE, 2u);

    uint32_t n_features;
    if (encoding.contains(EncodingParams::N_FEATURES)) {
        n_features = encoding[EncodingParams::N_FEATURES];
        if (encoding.contains(EncodingParams::N_LEVELS)) {
            throw std::runtime_error{"GridEncoding: may not specify n_features and n_levels "
                                     "simultaneously (one determines the other)"};
        }
    } else {
        n_features = N_FEATURES_PER_LEVEL * encoding.value(EncodingParams::N_LEVELS, 16u);
    }

    const uint32_t n_levels = n_features / N_FEATURES_PER_LEVEL;
    const GridType grid_type = encoding.value(EncodingParams::GRID_TYPE, GridType::Hash);

    const uint32_t base_resolution = encoding.value(EncodingParams::BASE_RESOLUTION, 16u);

#define TCNN_GRID_PARAMS                                                                                               \
    padded_output_width.has_value() ? padded_output_width.value() : n_features,                                        \
    n_features, log2_hashmap_size, base_resolution,                                                                    \
        encoding.value(EncodingParams::PER_LEVEL_SCALE,                                                                \
                       grid_type == GridType::Dense                                                                    \
                           ? std::exp(std::log(256.0f / (float)base_resolution) / (n_levels - 1))                      \
                           : 2.0f),                                                                                    \
        encoding.value(EncodingParams::USE_STOCHASTIC_INTERPOLATION, false),                                           \
        encoding.value(EncodingParams::INTERPOLATION_METHOD, InterpolationType::Linear), grid_type, Q

    // If higher-dimensional hash encodings are desired, corresponding switch
    // cases can be added
    switch (n_dims_to_encode) {
    case 2:
        return std::make_shared<GridEncodingTemplated<T, 2, N_FEATURES_PER_LEVEL, HASH_TYPE>>(TCNN_GRID_PARAMS);
    case 3:
        return std::make_shared<GridEncodingTemplated<T, 3, N_FEATURES_PER_LEVEL, HASH_TYPE>>(TCNN_GRID_PARAMS);
    default:
        throw std::runtime_error{"GridEncoding: number of input dims must be 2 or 3."};
    }
#undef TCNN_GRID_PARAMS
}

template <typename T, HashType HASH_TYPE>
std::shared_ptr<GridEncoding<T>> create_grid_encoding_templated_1(const json &encoding_config, 
    std::optional<uint32_t> padded_output_width, sycl::queue& Q) 
{
    const uint32_t n_features_per_level = encoding_config.value(EncodingParams::N_FEATURES_PER_LEVEL, 2u);
    switch (n_features_per_level) {
    case 1:
        return create_grid_encoding_templated_2<T, 1, HASH_TYPE>(encoding_config, padded_output_width, Q);
    case 2:
        return create_grid_encoding_templated_2<T, 2, HASH_TYPE>(encoding_config, padded_output_width, Q);
    case 4:
        return create_grid_encoding_templated_2<T, 4, HASH_TYPE>(encoding_config, padded_output_width, Q);
    case 8:
        return create_grid_encoding_templated_2<T, 8, HASH_TYPE>(encoding_config, padded_output_width, Q);
    default:
        throw std::runtime_error{"GridEncoding: n_features_per_level must be 1, 2, 4, or 8."};
    }
}

template <typename T> std::shared_ptr<GridEncoding<T>> create_grid_encoding(const json &encoding_config, 
    std::optional<uint32_t> padded_output_width, sycl::queue& Q) 
{
    const HashType hash_type = encoding_config.value(EncodingParams::HASH, HashType::CoherentPrime);

    switch (hash_type) {
    case HashType::Prime:
        return create_grid_encoding_templated_1<T, HashType::Prime>(encoding_config, padded_output_width, Q);
    case HashType::CoherentPrime:
        return create_grid_encoding_templated_1<T, HashType::CoherentPrime>(encoding_config, padded_output_width, Q);
    case HashType::ReversedPrime:
        return create_grid_encoding_templated_1<T, HashType::ReversedPrime>(encoding_config, padded_output_width, Q);
    case HashType::Rng:
        return create_grid_encoding_templated_1<T, HashType::Rng>(encoding_config, padded_output_width, Q);
    default:
        throw std::runtime_error{"GridEncoding: invalid hash type."};
    }
}

} // namespace grid
} // namespace encodings
} // namespace tinydpcppnn