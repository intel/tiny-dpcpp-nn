/**
 * @file kernel_helper.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Old sycl joint_matrix implementation helpers which does not work anymore.
 * TODO: remove this.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <array>
#include <sycl/sycl.hpp>

#include "common.h"

namespace tinydpcppnn {
namespace kernels {
namespace helpers {

using namespace sycl::ext::oneapi::experimental::matrix;

// load a submatrix row-major piece of size MxN int SLM
template <int M, int N, typename Tsrc, typename Tdest, sycl::access::address_space AddressSpacesrc,
          sycl::access::decorated IsDecoratedsrc, sycl::access::address_space AddressSpacedest,
          sycl::access::decorated IsDecorateddest>
static inline void moveMemory(sycl::nd_item<1> &item, const sycl::multi_ptr<Tsrc, AddressSpacesrc, IsDecoratedsrc> &src,
                              sycl::multi_ptr<Tdest, AddressSpacedest, IsDecorateddest> dest) {

    if constexpr (sizeof(Tdest) == 4 && sizeof(Tsrc) == 4)
        for (int iter = item.get_local_linear_id(); iter < M * N; iter += item.get_local_range(0)) {
            item.get_sub_group().store(dest + iter, static_cast<Tdest>(item.get_sub_group().load(src + iter)));
        }
    else if constexpr (sizeof(Tdest) == 2 && sizeof(Tsrc) == 2)
        for (int iter = item.get_local_linear_id(); iter < M * N; iter += 2 * item.get_local_range(0)) {
            item.get_sub_group().store(
                address_space_cast<AddressSpacedest, IsDecorateddest>((uint32_t *)&dest[iter]),
                item.get_sub_group().load(address_space_cast<AddressSpacesrc, IsDecoratedsrc>((uint32_t *)&src[iter])));
        }
}

// load a submatrix row-major piece of size MxN int SLM, sub-group by sub-group
template <int M, int N, typename Tsrc, typename Tdest, typename Group, sycl::access::address_space AddressSpacesrc,
          sycl::access::decorated IsDecoratedsrc, sycl::access::address_space AddressSpacedest,
          sycl::access::decorated IsDecorateddest>
static inline void moveMemorySG(Group sg, const sycl::multi_ptr<Tsrc, AddressSpacesrc, IsDecoratedsrc> &src,
                                sycl::multi_ptr<Tdest, AddressSpacedest, IsDecorateddest> dest) {

    if constexpr (sizeof(Tdest) == 4 && sizeof(Tsrc) == 4)
        for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()[0]) {
            dest[iter] = static_cast<Tdest>(src[iter]);
        }
    else if constexpr (sizeof(Tdest) == 2 && sizeof(Tsrc) == 2)
        for (int iter = sg.get_local_id()[0]; iter < M * N; iter += 2 * sg.get_local_range()[0]) {
            sg.store(address_space_cast<AddressSpacedest, IsDecorateddest>((uint32_t *)&dest[iter]),
                     sg.load(address_space_cast<AddressSpacesrc, IsDecoratedsrc>((uint32_t *)&src[iter])));
        }
}

template <int WIDTH, typename T, typename Group, sycl::access::address_space AddressSpace,
          sycl::access::decorated IsDecorated, use Use, layout Layout, size_t nMats, size_t TM, size_t TN>
static inline void moveMemorySG(Group sg, sycl::multi_ptr<T, AddressSpace, IsDecorated> src,
                                std::array<joint_matrix<Group, T, Use, TM, TN, Layout>, nMats> &mDest) {

    static_assert(nMats * TN == WIDTH);
    static_assert(Layout == layout::row_major);
    for (int iter = 0; iter < nMats; iter++) {
        joint_matrix_load(sg, mDest[iter], src + iter * TN, WIDTH);
    }
}

template <int WIDTH, typename Tsrc, typename Tdest, typename Group, sycl::access::address_space AddressSpace,
          sycl::access::decorated IsDecorated, use Use, layout Layout, size_t nMats, size_t TM, size_t TN>
static inline void moveMemorySG(Group sg, const std::array<joint_matrix<Group, Tsrc, Use, TM, TN, Layout>, nMats> &mSrc,
                                sycl::multi_ptr<Tdest, AddressSpace, IsDecorated> dest) {

    static_assert(nMats * TN == WIDTH);
    for (int iter = 0; iter < nMats; iter++) {
        sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, mSrc[iter], dest + iter * TN, WIDTH);
    }
}

template <typename Group, typename T, use Use, size_t NumRows, size_t NumCols, layout Layout, size_t Nmats>
static inline void zeroMatrices(Group sg,
                                std::array<joint_matrix<Group, T, Use, NumRows, NumCols, Layout>, Nmats> &matrices) {
#pragma unroll
    for (auto &mat : matrices) {
        joint_matrix_fill(sg, mat, static_cast<T>(0));
    }
}

template <size_t K, typename Group, typename Ta, typename Tb, typename Tc, sycl::access::address_space AddressSpaceA,
          sycl::access::decorated IsDecoratedA, sycl::access::address_space AddressSpaceB,
          sycl::access::decorated IsDecoratedB, size_t M, size_t N, size_t nCs>
static inline void MAD_1_ROW(Group sg, const sycl::multi_ptr<Ta, AddressSpaceA, IsDecoratedA> &A,
                             const sycl::multi_ptr<Tb, AddressSpaceB, IsDecoratedB> &B,
                             std::array<joint_matrix<Group, Tc, use::accumulator, M, N>, nCs> &mCs) {

    // WIDTH = nCs*N
    //  A is not vnnied
    joint_matrix<Group, Ta, use::a, M, K, layout::row_major> mA;
    joint_matrix<Group, Tb, use::b, K, N, layout::ext_intel_packed> mB;
    joint_matrix_load(sg, mA, A, nCs * N);
#pragma unroll
    for (int iter = 0; iter < nCs; iter++) {
        constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(Tb));
        joint_matrix_load(sg, mB, B + iter * vnni_factor * N, vnni_factor * nCs * N);
        joint_matrix_mad(sg, mCs[iter], mA, mB, mCs[iter]);
    }
}

template <size_t K, typename Group, typename Ta, typename Tb, typename Tc, sycl::access::address_space AddressSpaceA,
          sycl::access::decorated IsDecoratedA, sycl::access::address_space AddressSpaceB,
          sycl::access::decorated IsDecoratedB, size_t M, size_t N, size_t nCs>
static inline void MAD(Group sg, const sycl::multi_ptr<Ta, AddressSpaceA, IsDecoratedA> &A,
                       const sycl::multi_ptr<Tb, AddressSpaceB, IsDecoratedB> &B,
                       std::array<joint_matrix<Group, Tc, use::accumulator, M, N>, nCs> &mCs) {

    // WIDTH = nCs*N
    for (int aiter = 0; aiter < nCs * N; aiter += K) {
        MAD_1_ROW<K>(sg, A + aiter, B + aiter * nCs * N, mCs);
    }
}

template <size_t K, typename Group, typename Ta, typename Tb, typename Tc, layout LayoutA,
          sycl::access::address_space AddressSpaceB, sycl::access::decorated IsDecoratedB, size_t M, size_t N,
          size_t nAs, size_t nCs>
static inline void MAD(Group sg, const std::array<joint_matrix<Group, Ta, use::a, M, K, LayoutA>, nAs> &mAs,
                       const sycl::multi_ptr<Tb, AddressSpaceB, IsDecoratedB> &B,
                       std::array<joint_matrix<Group, Tc, use::accumulator, M, N>, nCs> &mCs) {

    // WIDTH = nCs*N
    //  A is not vnnied

    constexpr int vnni_factor_times_N = std::max<int>(1, 4 / sizeof(Tb)) * N;
    constexpr int offset_row = vnni_factor_times_N * nCs;

    // #pragma collapse 2 unroll
    for (int iterA = 0; iterA < nAs; iterA++) {
        for (int iter = 0; iter < nCs; iter++) {

            joint_matrix<Group, Tb, use::b, K, N, layout::ext_intel_packed> mB;
            joint_matrix_load(sg, mB, B + iter * vnni_factor_times_N + iterA * N * nCs * M, offset_row);
            joint_matrix_mad(sg, mCs[iter], mAs[iterA], mB, mCs[iter]);
        }
    }
}

template <typename Tin, typename Tout, Activation act> inline void activate(const Tin &data_in, Tout &data_out) {
    if constexpr (act == Activation::None)
        data_out = static_cast<Tout>(data_in);
    else if constexpr (act == Activation::ReLU)
        data_out = data_in > static_cast<Tin>(0) ? static_cast<Tout>(data_in) : static_cast<Tout>(0);
    else if constexpr (act == Activation::Tanh)
        data_out = static_cast<Tout>(std::tanh(float(data_in)));
}

template <typename Tin, typename Tdec, typename Tout, Activation act, sycl::access::address_space AddressSpace,
          sycl::access::decorated IsDecorated>
inline void activateBackward(const Tin &data_in, const sycl::multi_ptr<Tdec, AddressSpace, IsDecorated> data_decision,
                             Tout &data_out) {
    if constexpr (act == Activation::None)
        data_out = static_cast<Tout>(data_in);
    else if constexpr (act == Activation::ReLU)
        data_out = static_cast<Tout>(data_decision[0] > static_cast<Tdec>(0) ? data_in : 0);
}

/// TODO: generalize this in case in and dest dimensions do not coincide. If that is the case, go over SLM... or
/// something like that
template <Activation act, typename Group, use UseIn, use UseOut, typename Tin, typename Tout, size_t NumRows,
          size_t NumCols, layout LayoutIn, layout LayoutOut, size_t nMats>
static inline void
applyActivation(Group sg, std::array<joint_matrix<Group, Tin, UseIn, NumRows, NumCols, LayoutIn>, nMats> &in,
                std::array<joint_matrix<Group, Tout, UseOut, NumRows, NumCols, LayoutOut>, nMats> &dest) {

    // WIDTH = NumCols*nMats;
    for (auto matiter = 0; matiter < nMats; matiter++) {
        auto data_in = sycl::ext::oneapi::detail::get_wi_data(sg, in[matiter]);
        auto data_out = sycl::ext::oneapi::detail::get_wi_data(sg, dest[matiter]);
        for (int rowiter = 0; rowiter < data_in.length(); rowiter++) {
            Tout tmp;
            activate<Tin, Tout, act>(static_cast<Tin>(data_in[rowiter]), tmp);
            data_out[rowiter] = tmp;
        }
    }
}

template <Activation act, typename Group, use Use, typename Tin, typename Tout, size_t NumRows, size_t NumCols,
          layout Layout, sycl::access::address_space AddressSpace, sycl::access::decorated IsDecorated, size_t nMats>
static inline void applyActivation(Group sg,
                                   std::array<joint_matrix<Group, Tin, Use, NumRows, NumCols, Layout>, nMats> &in,
                                   sycl::multi_ptr<Tout, AddressSpace, IsDecorated> dest) {

    // WIDTH = NumCols*nMats;
    for (auto matiter = 0; matiter < nMats; matiter++) {
        auto data_in = sycl::ext::oneapi::detail::get_wi_data(sg, in[matiter]);
        for (int rowiter = 0; rowiter < data_in.length(); rowiter++) {
            activate<Tin, Tout, act>(static_cast<Tin>(data_in[rowiter]),
                                     dest[rowiter * NumCols * nMats + matiter * NumCols + sg.get_local_id()[0]]);
        }
    }
}

template <Activation act, int M, int N, typename Group, typename Tin, typename Tout,
          sycl::access::address_space AddressSpacesrc, sycl::access::decorated IsDecoratedsrc,
          sycl::access::address_space AddressSpacedest, sycl::access::decorated IsDecorateddest>
static inline void applyActivation(Group sg, const sycl::multi_ptr<Tin, AddressSpacesrc, IsDecoratedsrc> &src,
                                   sycl::multi_ptr<Tout, AddressSpacedest, IsDecorateddest> dest) {

    for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()[0]) {
        activate<Tin, Tout, act>(static_cast<Tin>(src[iter]), dest[iter]);
    }
}

template <Activation act, typename Group, use Use, typename Tin, typename Tdec, typename Tout, size_t NumRows,
          size_t NumCols, layout Layout, sycl::access::address_space AddressSpacedecision,
          sycl::access::decorated IsDecorateddecision, sycl::access::address_space AddressSpace,
          sycl::access::decorated IsDecorated, size_t nMats>
static inline void
applyBackwardActivation(Group sg, std::array<joint_matrix<Group, Tin, Use, NumRows, NumCols, Layout>, nMats> &in,
                        const sycl::multi_ptr<Tdec, AddressSpacedecision, IsDecorateddecision> decision_values,
                        sycl::multi_ptr<Tout, AddressSpace, IsDecorated> dest) {

    // WIDTH = NumCols*nMats;
    for (auto matiter = 0; matiter < nMats; matiter++) {
        auto data_in = sycl::ext::oneapi::detail::get_wi_data(sg, in[matiter]);
        for (int rowiter = 0; rowiter < data_in.length(); rowiter++) {
            const size_t offset = rowiter * NumCols * nMats + matiter * NumCols + sg.get_local_id()[0];
            activateBackward<Tin, Tdec, Tout, act>(static_cast<Tin>(data_in[rowiter]), decision_values + offset,
                                                   dest[offset]);
        }
    }
}

template <Activation act, int M, int N, typename Group, typename Tin, typename Tdec, typename Tout,
          sycl::access::address_space AddressSpacesrc, sycl::access::decorated IsDecoratedsrc,
          sycl::access::address_space AddressSpacedecision, sycl::access::decorated IsDecorateddecision,
          sycl::access::address_space AddressSpacedest, sycl::access::decorated IsDecorateddest>
static inline void
applyBackwardActivation(Group sg, const sycl::multi_ptr<Tin, AddressSpacesrc, IsDecoratedsrc> &src,
                        const sycl::multi_ptr<Tout, AddressSpacedecision, IsDecorateddecision> decision_values,
                        sycl::multi_ptr<Tout, AddressSpacedest, IsDecorateddest> dest) {

    for (int iter = sg.get_local_id()[0]; iter < M * N; iter += sg.get_local_range()[0]) {
        activateBackward<Tin, Tdec, Tout, act>(static_cast<Tin>(src[iter]), decision_values + iter, dest[iter]);
    }
}

} // namespace helpers
} // namespace kernels
} // namespace tinydpcppnn