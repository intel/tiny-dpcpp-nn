/**
 * @file kernel.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Old SYCL joint_matrix kenrel function implementation.
 * TODO: remove this.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <algorithm>
#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"
// #include "kernel_helper.h"

namespace tinydpcppnn {
namespace kernels {

enum LSC_LDCC {
    LSC_LDCC_DEFAULT,
    LSC_LDCC_L1UC_L3UC, // 1 // Override to L1 uncached and L3 uncached
    LSC_LDCC_L1UC_L3C,  // 2 // Override to L1 uncached and L3 cached
    LSC_LDCC_L1C_L3UC,  // 3 // Override to L1 cached and L3 uncached
    LSC_LDCC_L1C_L3C,   // 4 // Override to L1 cached and L3 cached
    LSC_LDCC_L1S_L3UC,  // 5 // Override to L1 streaming load and L3 uncached
    LSC_LDCC_L1S_L3C,   // 6 // Override to L1 streaming load and L3 cached
    LSC_LDCC_L1IAR_L3C, // 7 // Override to L1 invalidate-after-read, and L3
                        // cached
};

extern "C" {
SYCL_EXTERNAL void __builtin_IB_lsc_prefetch_global_uchar(const __attribute__((opencl_global)) uint8_t *base,
                                                          int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void __builtin_IB_lsc_prefetch_global_ushort(const __attribute__((opencl_global)) uint16_t *base,
                                                           int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void __builtin_IB_lsc_prefetch_global_uint(const __attribute__((opencl_global)) uint32_t *base,
                                                         int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void __builtin_IB_lsc_prefetch_global_uint2(const __attribute__((opencl_global)) uint32_t *base,
                                                          int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void __builtin_IB_lsc_prefetch_global_uint3(const __attribute__((opencl_global)) uint32_t *base,
                                                          int immElemOff, enum LSC_LDCC cacheOpt);
SYCL_EXTERNAL void __builtin_IB_lsc_prefetch_global_uint4(const __attribute__((opencl_global)) uint32_t *base,
                                                          int immElemOff, enum LSC_LDCC cacheOpt);
}

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, size_t TN>
inline std::vector<sycl::event>
batchedGEMM_naive(sycl::queue &q, T *const __restrict__ output_ptr, T const *const __restrict__ intermediate_forward,
                  T const *const __restrict__ intermediate_backward, const int n_hidden_layers, const int M,
                  const std::vector<sycl::event> &deps) {
    constexpr int SG_SIZE = TN;
    auto e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_hidden_layers + 1, WIDTH * WIDTH),
                                           sycl::range<2>(1, std::min(1024, WIDTH * WIDTH))),
                         [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                             const int matrix = item.get_global_id(0);
                             const int element = item.get_global_id(1);
                             const int row = element / WIDTH;
                             const int col = element % WIDTH;

                             Tc tmp_out = static_cast<Tc>(0);
                             T const *intermediate_forward_loc = intermediate_forward + matrix * M * WIDTH + row;
                             T const *intermediate_backward_loc = intermediate_backward + matrix * M * WIDTH + col;
                             for (int i = 0; i < M; i++) {
                                 tmp_out += static_cast<Tc>(*intermediate_forward_loc) *
                                            static_cast<Tc>(*intermediate_backward_loc);
                                 intermediate_forward_loc += WIDTH;
                                 intermediate_backward_loc += WIDTH;
                             }
                             T *const output_ptr_loc = output_ptr + WIDTH * WIDTH * matrix + element;
                             *output_ptr_loc = static_cast<T>(tmp_out);
                         });
    });
    // auto e =
    //     q.parallel_for((n_hidden_layers + 1) * WIDTH * WIDTH, [=](auto item) [[intel::reqd_sub_group_size(SG_SIZE)]]
    //     {
    //         output_ptr[item.get_id()] = static_cast<T>(1.23);
    //     });

    return {e};
}

////////////////////////////GENERAL FUNCTIONS WHICH CAN DO EVERYTHING///////////

// Todo: May want to remove some of the template parameters of these functions and
// make them inputs.

// This is the general forward map which also doubles as inference. We use template
// specialization for all the versions
template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, bool INFERENCE, size_t TN>
std::vector<sycl::event> forward_impl_general(sycl::queue &q, T const *const __restrict__ weights_ptr,
                                              T const *const __restrict__ inputs_ptr,
                                              T *const __restrict__ intermediate_output, const int n_hidden_layers,
                                              const int M, const std::vector<sycl::event> &deps) {

    throw std::logic_error("General function should not be called.");
    /*    static_assert(INPUT_WIDTH == WIDTH);
        static_assert(OUTPUT_WIDTH == WIDTH);
        static_assert(WIDTH % TN == 0);

        constexpr int SG_SIZE = TN;
        constexpr size_t TM = 8;
        assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses // this may be adjusted in
       the
                             // future
        constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T))); // This depends on the datatype T
        int SGS_IN_WG = std::min(M / TM, q.get_device().get_info<sycl::info::device::max_work_group_size>() / SG_SIZE);
        while (M / TM % SGS_IN_WG != 0) {
            SGS_IN_WG--;
        }
        if (SGS_IN_WG <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");
        constexpr int NC = WIDTH / TN; // number of systolic C matrices in the output

        // One Block Row has TM rows an N columns.
        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);

            sycl::local_accessor<T, 1> B(sycl::range<1>(WIDTH * WIDTH), cgh);
            sycl::local_accessor<T, 1> Atmp(sycl::range<1>(TM * WIDTH * SGS_IN_WG), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                    auto sg = item.get_sub_group();

                    auto weights_ptr_loc =
                        address_space_cast<access::address_space::global_space, access::decorated::yes>(weights_ptr);
                    auto intermediate_output_loc =
                        address_space_cast<access::address_space::global_space, access::decorated::yes>(
                            intermediate_output);
                    auto A_sg_start =
                        Atmp.template get_multi_ptr<access::decorated::yes>() + sg.get_group_id()[0] * WIDTH * TM;
                    auto B_ptr = B.template get_multi_ptr<access::decorated::yes>();

                    // offset in all the data
                    const int wg_and_sg_offset_A =
                        item.get_group().get_group_id() * SGS_IN_WG * WIDTH * TM + sg.get_group_id()[0] * WIDTH * TM;
                    int layer_offset_A = M * WIDTH + wg_and_sg_offset_A;

                    helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                    weights_ptr_loc += WIDTH * WIDTH; // next weight matrix

                    // load input in slm
                    helpers::moveMemorySG<TM, WIDTH>(
                        sg,
                        address_space_cast<access::address_space::global_space, access::decorated::yes>(inputs_ptr +
                                                                                                        wg_and_sg_offset_A),
                        A_sg_start);

                    // if not inference activate and store in intermediate output
                    if constexpr (!INFERENCE)
                        helpers::applyActivation<activation, TM, WIDTH>(sg, A_sg_start,
                                                                        intermediate_output_loc + wg_and_sg_offset_A);

                    std::array<joint_matrix<sycl::sub_group, Tc, use::accumulator, TM, TN>, NC> Cs;
                    for (int layer = 0; layer < n_hidden_layers; layer++) {
                        // reset result matrices
                        helpers::zeroMatrices(sg, Cs);

                        // ensure weight matrix is loaded
                        item.barrier(sycl::access::fence_space::local_space);

                        helpers::MAD<TK>(sg, A_sg_start, B_ptr, Cs);

                        item.barrier(sycl::access::fence_space::local_space);
                        // load next weight matrix

                        helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                        weights_ptr_loc += WIDTH * WIDTH; // next weight matrix

                        // activate and save
                        helpers::applyActivation<activation>(sg, Cs, A_sg_start);

                        if constexpr (!INFERENCE)
                            helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);

                        layer_offset_A += M * WIDTH;
                    }

                    // generate output, i.e. last GEMM
                    helpers::zeroMatrices(sg, Cs);

                    // wait for B to be loaded
                    item.barrier(sycl::access::fence_space::local_space);

                    helpers::MAD<TK>(sg, A_sg_start, B_ptr, Cs);

                    // activate and save to slm
                    helpers::applyActivation<output_activation>(sg, Cs, A_sg_start);

                    // save slm to HBM
                    helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);
                });
        });

        return {e};*/
}

// template <>
// std::vector<sycl::event> forward_impl_general<bf16, float, 64, 64, 64, Activation::ReLU, Activation::None, true, 16>(
//     sycl::queue &q, bf16 const *const __restrict__ weights_ptr, bf16 const *const __restrict__ inputs_ptr,
//     bf16 *const __restrict__ intermediate_output, const int n_hidden_layers, const int M,
//     const std::vector<sycl::event> &deps) {
//     // Indicates how many joint_matrix rows (i.e. time TM actual rows) are done by one
//     // sub-group. ONLY works for = 1 right now.
//     // reuse of B, this is in subgroups, ONLY works for 64 nor
//     // note that large grf mode requires this to be set to 32
//     constexpr int SGS_IN_WG = 64;
//     constexpr int N = 64;
//     constexpr int K = 64;
//     constexpr int TM = 8;
//     constexpr int TN = 16;
//     constexpr int TK = 16;
//     constexpr int SG_SIZE = 16;

//     // One Block Row has TM rows an N columns.
//     auto e = q.submit([&](handler &cgh) {
//         cgh.depends_on(deps);
//         local_accessor<bf16, 1> B(range<1>(K * N),
//                                   cgh); // weights matrix. 64*64*2 byte = 8 kb. Thus, can have up to 16 WGs per Xe
//                                   Core.
//         local_accessor<bf16, 1> Atmp(range<1>(TM * K * SGS_IN_WG),
//                                      cgh); // buffer for loading joint matrices. 8*64*64*2byte = 64kb. TODO: check if
//                                            // this is too much. If so, split in half
//         // number of SGS is given by batch_size / TM, since batch_size is the number of rows in the output

//         cgh.parallel_for(
//             nd_range<1>(std::max(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
//                         SGS_IN_WG * SG_SIZE), // assuming here that the number of block rows is divisable by
//                         SGS_IN_WG
//             [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
//                 const int wg_id = item.get_group().get_group_id();
//                 auto sg = item.get_sub_group();
//                 const int loc_id = sg.get_local_id()[0];
//                 const int sg_id = sg.get_group_id()[0];

//                 bf16 const *weights_ptr_loc = weights_ptr;

//                 const int sg_offset_B = sg_id * N * K / SGS_IN_WG; // we assume this is divisible
//                 const int sg_offset_A = sg_id * K * TM;
//                 const int total_offset_A = wg_id * SGS_IN_WG * K * TM + sg_offset_A;

//                 // Load B into slm
//                 ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id];
//                 ((int32_t *)(&B[sg_offset_B]))[loc_id + SG_SIZE] =
//                     ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id + SG_SIZE];
//                 weights_ptr_loc += K * N;

//                 sycl::vec<int32_t, TM> tmp16avalues0, tmp16avalues1;
//                 for (int iter = 0; iter < TM; iter++) {
//                     tmp16avalues0[iter] = *((int32_t *)(inputs_ptr + total_offset_A) + loc_id + iter * K / 2);
//                     tmp16avalues1[iter] = *((int32_t *)(inputs_ptr + total_offset_A) + loc_id + iter * K / 2 +
//                     SG_SIZE);
//                 }

//                 for (int iter = 0; iter < TM; iter++) {
//                     *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2) = tmp16avalues0[iter];
//                     *(((int32_t *)&Atmp[sg_offset_A]) + loc_id + iter * K / 2 + SG_SIZE) = tmp16avalues1[iter];
//                 }

//                 joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> A_block0, A_block1, A_block2,
//                 A_block3;

//                 joint_matrix_load(sg, A_block0, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 0 * SG_SIZE, K);
//                 joint_matrix_load(sg, A_block1, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 1 * SG_SIZE, K);
//                 joint_matrix_load(sg, A_block2, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 2 * SG_SIZE, K);
//                 joint_matrix_load(sg, A_block3, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 3 * SG_SIZE, K);

//                 joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed>
//                     B_block;
//                 joint_matrix<sub_group, float, use::accumulator, TM, TN> C_block0, C_block1, C_block2, C_block3;

//                 for (int layer = 0; layer < n_hidden_layers; layer++) {
//                     // reset result matrix

//                     joint_matrix_fill(sg, C_block0, 0.0f);
//                     joint_matrix_fill(sg, C_block1, 0.0f);
//                     joint_matrix_fill(sg, C_block2, 0.0f);
//                     joint_matrix_fill(sg, C_block3, 0.0f);

//                     item.barrier(sycl::access::fence_space::local_space);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 0 * 2 * TN, 2 * N);
//                     C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 1 * 2 * TN, 2 * N);
//                     C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 2 * 2 * TN, 2 * N);
//                     C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 3 * 2 * TN, 2 * N);
//                     C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 0 * 2 * TN, 2 * N);
//                     C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 1 * 2 * TN, 2 * N);
//                     C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 2 * 2 * TN, 2 * N);
//                     C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 3 * 2 * TN, 2 * N);
//                     C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 0 * 2 * TN, 2 * N);
//                     C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 1 * 2 * TN, 2 * N);
//                     C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 2 * 2 * TN, 2 * N);
//                     C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 3 * 2 * TN, 2 * N);
//                     C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 0 * 2 * TN, 2 * N);
//                     C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 1 * 2 * TN, 2 * N);
//                     C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 2 * 2 * TN, 2 * N);
//                     C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 3 * 2 * TN, 2 * N);
//                     C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);

//                     // Load next B into slm

//                     item.barrier(sycl::access::fence_space::local_space); // make sure all the reads are done before
//                     we
//                                                                           // write into slm again
//                     ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id];
//                     ((int32_t *)(&B[sg_offset_B]))[loc_id + SG_SIZE] =
//                         ((int32_t *)(weights_ptr_loc + sg_offset_B))[loc_id + SG_SIZE];
//                     weights_ptr_loc += K * N;

//                     // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.

//                     auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
//                     auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
//                     auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
//                     auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
//                     auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
//                     auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
//                     auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
//                     auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);

//                     for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
//                     {
//                         Ai_data0[rowiter] = fmax((bf16)0, (bf16)Ci_data0[rowiter]); // tmpCi < (bf16)0 ? (bf16)0 :
//                                                                                     // tmpCi;
//                         Ai_data1[rowiter] = fmax((bf16)0, (bf16)Ci_data1[rowiter]);
//                         Ai_data2[rowiter] = fmax((bf16)0, (bf16)Ci_data2[rowiter]);
//                         Ai_data3[rowiter] = fmax((bf16)0, (bf16)Ci_data3[rowiter]);
//                     }
//                 }

//                 joint_matrix_fill(sg, C_block0, 0.0f);
//                 joint_matrix_fill(sg, C_block1, 0.0f);
//                 joint_matrix_fill(sg, C_block2, 0.0f);
//                 joint_matrix_fill(sg, C_block3, 0.0f);

//                 item.barrier(sycl::access::fence_space::local_space); // wait for B to be loaded

//                 // block axpy scheme
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 0 * 2 * TN, 2 * N);
//                 C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 1 * 2 * TN, 2 * N);
//                 C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 2 * 2 * TN, 2 * N);
//                 C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * N * TK + 3 * 2 * TN, 2 * N);
//                 C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 0 * 2 * TN, 2 * N);
//                 C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 1 * 2 * TN, 2 * N);
//                 C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 2 * 2 * TN, 2 * N);
//                 C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * N * TK + 3 * 2 * TN, 2 * N);
//                 C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 0 * 2 * TN, 2 * N);
//                 C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 1 * 2 * TN, 2 * N);
//                 C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 2 * 2 * TN, 2 * N);
//                 C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * N * TK + 3 * 2 * TN, 2 * N);
//                 C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 0 * 2 * TN, 2 * N);
//                 C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 1 * 2 * TN, 2 * N);
//                 C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 2 * 2 * TN, 2 * N);
//                 C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * N * TK + 3 * 2 * TN, 2 * N);
//                 C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);

//                 /// TODO: Output activation in what follows. Here == None
//                 // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
//                 // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
//                 auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
//                 auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
//                 auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
//                 auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
//                 auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
//                 auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
//                 auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
//                 auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);

//                 for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
//                 {
//                     Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
//                     Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
//                     Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
//                     Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
//                 }

//                 const int loc_offset_A = (n_hidden_layers + 1) * M * K + total_offset_A;
//                 // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block0, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 0 * SG_SIZE, K);
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block1, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 1 * SG_SIZE, K);
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block2, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 2 * SG_SIZE, K);
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block3, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 3 * SG_SIZE, K);
//                 /// Alternative of loading A through SLM to avoid inefficient access to HBM
//                 // we do not need SLM barrier since each SG writes and reads only its own data.

//                 for (int iter = 0; iter < TM; iter++) {
//                     *((int32_t *)(intermediate_output + loc_offset_A + iter * K) + loc_id) =
//                         *(((int32_t *)&Atmp[sg_offset_A + iter * K]) + loc_id);
//                     *((int32_t *)(intermediate_output + loc_offset_A + iter * K) + loc_id + SG_SIZE) =
//                         *(((int32_t *)&Atmp[sg_offset_A + iter * K]) + loc_id + SG_SIZE);
//                 }
//             });
//     });

//     return {e};
// }

template <typename T, typename Tc, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation,
          Activation output_activation, size_t TN>
std::vector<sycl::event> backward_impl_general(queue &q, T const *const __restrict__ weights_ptr,
                                               T const *const __restrict__ inputs_ptr, T *const __restrict__ output_ptr,
                                               T *const __restrict__ intermediate_output,
                                               T const *const __restrict__ forward, const int n_hidden_layers,
                                               const int M, const std::vector<sycl::event> &deps) {

    // make sure there is no remainder and no out of bounds accesses
    /*static_assert(WIDTH % TN == 0);
    // only works for input_width == width == output_width
    static_assert(INPUT_WIDTH == WIDTH);
    static_assert(OUTPUT_WIDTH == WIDTH);

    constexpr int SG_SIZE = TN;
    // this may be adjusted in the future in dpendence of M
    constexpr size_t TM = 8;
    int SGS_IN_WG = std::min(M / TM, q.get_device().get_info<sycl::info::device::max_work_group_size>() / SG_SIZE);
    /// TODO: say we use M/TM = 65. Then this results in WG=1 SG and too many slm load of B.
    /// Better: Use max size WGs and return those which are larger than M/TM. But
    /// requires special care for the loading of B
    while (M / TM % SGS_IN_WG != 0) {
        SGS_IN_WG--;
    }
    if (SGS_IN_WG <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");

    // number of systolic C matrices in the output
    constexpr int NC = WIDTH / TN;
    assert(M % TM == 0);
    // TK depends on the datatype T
    constexpr size_t TK = 8 * std::min<size_t>(8, 32 / (8 * sizeof(T)));

    auto e = q.submit([&](handler &cgh) {
        cgh.depends_on(deps);

        local_accessor<T, 1> B(range<1>(WIDTH * WIDTH), cgh);
        local_accessor<T, 1> Atmp(range<1>(TM * WIDTH * SGS_IN_WG), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
            [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();

                auto weights_ptr_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(weights_ptr) +
                    n_hidden_layers * WIDTH * WIDTH;
                auto intermediate_output_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(
                        intermediate_output);
                const auto forward_loc =
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(forward);
                auto A_sg_start =
                    Atmp.template get_multi_ptr<access::decorated::yes>() + sg.get_group_id()[0] * WIDTH * TM;
                auto B_ptr = B.template get_multi_ptr<access::decorated::yes>();

                // offset in all the data
                const int wg_and_sg_offset_A =
                    item.get_group().get_group_id() * SGS_IN_WG * WIDTH * TM + sg.get_group_id()[0] * WIDTH * TM;
                /// TODO: check if this is n_hidden_layers or n_hidden_layers+1
                int layer_offset_A = n_hidden_layers * M * WIDTH + wg_and_sg_offset_A;

                // Get B into slm
                helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                weights_ptr_loc -= WIDTH * WIDTH; // decrease weights pointer by one layer

                // load input in slm
                helpers::moveMemorySG<TM, WIDTH>(
                    sg,
                    address_space_cast<access::address_space::global_space, access::decorated::yes>(inputs_ptr +
                                                                                                    wg_and_sg_offset_A),
                    A_sg_start);

                // store backward activated input to the last intermediate output
                // note that output_activation == ReLU does not need any work since that means
                // forward >= 0
                if constexpr (output_activation != Activation::None && output_activation != Activation::ReLU) {
                    helpers::applyBackwardActivation<output_activation, TM, WIDTH>(
                        sg, A_sg_start, forward_loc + layer_offset_A + M * WIDTH, A_sg_start);
                }

                // store activated slm in intermediate output
                helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);

                std::array<joint_matrix<sycl::sub_group, Tc, use::accumulator, TM, TN>, NC> Cs;
                // we are also doing output->last hidden layer
                for (int layer = n_hidden_layers; layer > 0; layer--) {
                    layer_offset_A -= M * WIDTH;
                    helpers::zeroMatrices(sg, Cs);

                    // wait for B to be done storing
                    item.barrier(sycl::access::fence_space::local_space);

                    helpers::MAD<TK>(sg, A_sg_start, B_ptr, Cs);

                    // load B for next iteration into SLM
                    if (layer > 1) {
                        // wait for B to de done in the MAD
                        item.barrier(sycl::access::fence_space::local_space);
                        helpers::moveMemory<WIDTH, WIDTH>(item, weights_ptr_loc, B_ptr);
                        weights_ptr_loc -= WIDTH * WIDTH;
                    }

                    // If forward activation is ReLU we also do not need to do anything since all the values in forward
                    // are >= 0
                    helpers::applyBackwardActivation<activation == Activation::ReLU || activation == Activation::None
                                                         ? Activation::None
                                                         : activation>(sg, Cs, forward_loc + layer_offset_A + M * WIDTH,
                                                                       A_sg_start);

                    // store A slm to HBM
                    helpers::moveMemorySG<TM, WIDTH>(sg, A_sg_start, intermediate_output_loc + layer_offset_A);
                }
            });
    });

    // // NOTE: MKL gemm_batch is slower.
    // std::vector<sycl::event> events(n_hidden_layers + 1);
    // if constexpr (std::is_same<T, bf16>::value) { // need to cast to onemkls bf16 type.
    //     for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
    //         events[iter] = oneapi::mkl::blas::row_major::gemm(
    //             q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0f,
    //             reinterpret_cast<const oneapi::mkl::bfloat16 *>(forward) + iter * M * WIDTH, WIDTH,
    //             reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_output) + iter * M * WIDTH, WIDTH, 1.0f,
    //             reinterpret_cast<oneapi::mkl::bfloat16 *>(output_ptr) + iter * WIDTH * WIDTH, WIDTH, {e});
    //     }
    // } else {
    //     throw std::invalid_argument("Untested code path.");
    //     for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
    //         events[iter] = oneapi::mkl::blas::row_major::gemm(
    //             q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0,
    //             forward + iter * M * WIDTH, WIDTH, intermediate_output + iter * M * WIDTH, WIDTH, 1.0,
    //             output_ptr + iter * WIDTH * WIDTH, WIDTH, {e});
    //     }
    // }
    // return events;

    return batchedGEMM_naive<T, Tc, INPUT_WIDTH, WIDTH, OUTPUT_WIDTH, TN>(q, output_ptr, forward, intermediate_output,
                                                                          n_hidden_layers, M, {e});
                                                                          */
}

// // fused operation which does forward_pass+error computation + backward pass
// template <int WIDTH, Activation activation, Activation output_activation>
// std::vector<sycl::event> mlp_swift_fused(queue &q, bf16 const *const __restrict__ weights_ptr,
//                                          bf16 const *const __restrict__ weightsT_ptr,
//                                          bf16 const *const __restrict__ inputs_ptr,  // input to forward pass
//                                          bf16 const *const __restrict__ targets_ptr, // targets for error computation
//                                          bf16 *const __restrict__ output_ptr, // gradients output after backward pass
//                                          bf16 *const __restrict__ intermediate_output_forward,
//                                          bf16 *const __restrict__ intermediate_output_backward,
//                                          const int n_hidden_layers, const int M, const std::vector<sycl::event>
//                                          &deps) {
//     // reuse of B, this is in subgroups, ONLY works for 64 nor
//     // note that large grf mode requires this to be set to 32
//     constexpr int SGS_IN_WG = 64;
//     constexpr int TM = 8;
//     constexpr int TK = 16;
//     constexpr int TN = 16;
//     constexpr int SG_SIZE = 16;
//     // dimensions are M = batch_size, N = WIDTH = K = 64;
//     static_assert(TK == SG_SIZE);
//     static_assert(TN == TK);
//     if constexpr (WIDTH != 64) throw std::invalid_argument("Current implementation only works for a WIDTH of 64");
//     assert(M % TM == 0); // make sure there is no remainder and no out of bounds accesses
//     // Note that TN = TK = SG_SIZE

//     // One Block Row has TM rows an N columns.
//     auto e = q.submit([&](handler &cgh) {
//         cgh.depends_on(deps);
//         local_accessor<bf16, 1> B(range<1>(WIDTH * WIDTH),
//                                   cgh); // weights matrix. 64*64*2 byte = 8 kb. Thus, can have up to 16 WGs per Xe
//                                   Core.
//         local_accessor<float, 1> TmpOut(range<1>(WIDTH * WIDTH), cgh);
//         local_accessor<bf16, 1> Atmp(range<1>(TM * WIDTH * SGS_IN_WG),
//                                      cgh); // buffer for loading joint matrices. 8*64*64*2byte = 64kb. TODO: check if
//                                            // this is too much. If so, split in half
//         // number of SGS is given by batch_size / (TM), since batch_size is the number of rows in the output

//         cgh.parallel_for(
//             nd_range<1>(std::max(M / TM * SG_SIZE, SGS_IN_WG * SG_SIZE),
//                         SGS_IN_WG * SG_SIZE), // assuming here that the number of block rows is divisable by
//                         SGS_IN_WG
//             [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
//                 const int wg_id = item.get_group().get_group_id();
//                 auto sg = item.get_sub_group();
//                 const uint16_t loc_id = sg.get_local_id()[0]; // is in 0-15
//                 const uint16_t sg_id = sg.get_group_id()[0];  // is in 0-63

//                 /// Start with forward pass

//                 const uint16_t sg_offset_B =
//                     sg_id * WIDTH *
//                     (WIDTH / SGS_IN_WG); // we assume SGS_IN_WG divides K //is in the rang 0-64*64=0-4096
//                 const uint16_t sg_offset_A = sg_id * WIDTH * TM; // offset in WG is in the range 0-64*64*8=0-32K
//                 const int total_offset_A = wg_id * SGS_IN_WG * WIDTH * TM + sg_offset_A; // offset in current block
//                 __builtin_IB_lsc_prefetch_global_uint4(
//                     (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id, 0,
//                     LSC_LDCC_L1C_L3UC);
//                 __builtin_IB_lsc_prefetch_global_uint4(
//                     (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id +
//                         4 * SG_SIZE,
//                     0, LSC_LDCC_L1C_L3UC);
//                 __builtin_IB_lsc_prefetch_global_uint4(
//                     (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id +
//                         8 * SG_SIZE,
//                     0, LSC_LDCC_L1C_L3UC);
//                 __builtin_IB_lsc_prefetch_global_uint4(
//                     (const __attribute__((opencl_global)) uint32_t *)(inputs_ptr + total_offset_A) + loc_id +
//                         16 * SG_SIZE,
//                     0, LSC_LDCC_L1C_L3UC);

//                 // Load B into slm
//                 /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
//                 ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t *)(weights_ptr + sg_offset_B))[loc_id];
//                 ((int32_t *)(&B[sg_offset_B]))[loc_id + SG_SIZE] =
//                     ((int32_t *)(weights_ptr + sg_offset_B))[loc_id + SG_SIZE];

//                 // load input
//                 /// Alternative of loading A through SLM to avoid inefficient access to HBM
//                 sycl::vec<int32_t, TM> tmp16avalues0 =
//                     sg.load<8>(global_ptr<int32_t>((int32_t *)(inputs_ptr + total_offset_A)));
//                 sycl::vec<int32_t, TM> tmp16avalues1 =
//                     sg.load<8>(global_ptr<int32_t>((int32_t *)(inputs_ptr + total_offset_A + 4 * WIDTH)));

//                 sg.store<8>(local_ptr<int32_t>((int32_t *)&Atmp[sg_offset_A]), tmp16avalues0);
//                 sg.store<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A + 4 * WIDTH])), tmp16avalues1);
//                 // we do not need SLM barrier since each SG writes and reads only its own data.
//                 // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access

//                 // ATTENTION: current inputs are positive and this is not really tested
//                 if constexpr (activation == Activation::ReLU) {
//                     int32_t bitmask = 0b11111111111111110000000000000000;
//                     for (uint8_t iter = 0; iter < TM; iter++) {
//                         if ((tmp16avalues0[iter] >> 31) & 1) tmp16avalues0[iter] &= ~bitmask;
//                         if ((tmp16avalues0[iter] >> 15) & 1) tmp16avalues0[iter] &= bitmask;

//                         if ((tmp16avalues1[iter] >> 31) & 1) tmp16avalues1[iter] &= ~bitmask;
//                         if ((tmp16avalues1[iter] >> 15) & 1) tmp16avalues1[iter] &= bitmask;
//                     }
//                 }

//                 sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + total_offset_A)),
//                             tmp16avalues0);
//                 sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + total_offset_A + 4 *
//                 WIDTH)),
//                             tmp16avalues1);

//                 joint_matrix<sub_group, bf16, use::a, TM, TK, layout::row_major> A_block0, A_block1, A_block2,
//                 A_block3;

//                 joint_matrix_load(sg, A_block0, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 0 * SG_SIZE, WIDTH);
//                 joint_matrix_load(sg, A_block1, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 1 * SG_SIZE, WIDTH);
//                 joint_matrix_load(sg, A_block2, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 2 * SG_SIZE, WIDTH);
//                 joint_matrix_load(sg, A_block3, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 3 * SG_SIZE, WIDTH);

//                 // We have n_hidden_layers. Thus n_hidden_layers - 1 gemms between
//                 // the layers (layer 0 -> GEMM -> layer1 -> GEMM -> layer2 -> etc.)
//                 // Since we also do the GEMM from input to hidden layer 0,
//                 // we perform n_hidden_layers GEMMS.
//                 joint_matrix<sub_group, bf16, use::b, TK, TN, sycl::ext::intel::experimental::matrix::layout::packed>
//                     B_block;
//                 joint_matrix<sub_group, float, use::accumulator, TM, TN> C_block0, C_block1, C_block2, C_block3;

//                 for (uint8_t layer = 0; layer < n_hidden_layers; layer++) {

//                     joint_matrix_fill(sg, C_block0, 0.0f);
//                     joint_matrix_fill(sg, C_block1, 0.0f);
//                     joint_matrix_fill(sg, C_block2, 0.0f);
//                     joint_matrix_fill(sg, C_block3, 0.0f);

//                     item.barrier(sycl::access::fence_space::local_space); // wait for B to be loaded

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);

//                     // Load next B into slm
//                     /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
//                     item.barrier(sycl::access::fence_space::local_space);
//                     sg.store<2>(local_ptr<int32_t>((int32_t *)(&B[sg_offset_B])),
//                                 sg.load<2>(global_ptr<int32_t>(
//                                     (int32_t *)(weights_ptr + (layer + 1) * WIDTH * WIDTH + sg_offset_B))));

//                     // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.

//                     auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
//                     auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
//                     auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
//                     auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
//                     auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
//                     auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
//                     auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
//                     auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);

//                     for (int rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
//                     {
//                         Ai_data0[rowiter] = fmax((bf16)0, (bf16)Ci_data0[rowiter]); // tmpCi < (bf16)0 ? (bf16)0 :
//                                                                                     // tmpCi;
//                         Ai_data1[rowiter] = fmax((bf16)0, (bf16)Ci_data1[rowiter]);
//                         Ai_data2[rowiter] = fmax((bf16)0, (bf16)Ci_data2[rowiter]);
//                         Ai_data3[rowiter] = fmax((bf16)0, (bf16)Ci_data3[rowiter]);
//                     }

//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block0, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 0 * SG_SIZE, WIDTH);
//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block1, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 1 * SG_SIZE, WIDTH);
//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block2, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 2 * SG_SIZE, WIDTH);
//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block3, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 3 * SG_SIZE, WIDTH);
//                     /// Alternative of loading A through SLM to avoid inefficient access to HBM
//                     // we do not need SLM barrier since each SG writes and reads only its own data.

//                     const int loc_offset_A = (layer + 1) * M * WIDTH + total_offset_A;
//                     sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A)),
//                                 sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[0] + sg_offset_A))));
//                     sg.store<8>(
//                         global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A + 4 * WIDTH)),
//                         sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[0] + sg_offset_A + 4 * WIDTH))));
//                 }

//                 // generate output, i.e. last GEMM, differs since it uses output_activation
//                 // reset result matrix
//                 joint_matrix_fill(sg, C_block0, 0.0f);
//                 joint_matrix_fill(sg, C_block1, 0.0f);
//                 joint_matrix_fill(sg, C_block2, 0.0f);
//                 joint_matrix_fill(sg, C_block3, 0.0f);

//                 item.barrier(sycl::access::fence_space::local_space); // wait for B to be loaded

//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                 C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                 C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                 C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                 C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                 C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                 C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                 C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                 C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                 C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                 C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                 C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                 C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                 C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                 C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                 C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
//                 joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                 C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);

//                 /// TODO: Output activation in what follows. Here == None
//                 // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
//                 // This can be done in the future with joint_matrix_copy and a joint_maitrx_apply.
//                 auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
//                 auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
//                 auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
//                 auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
//                 auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
//                 auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
//                 auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
//                 auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);

//                 for (uint8_t rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
//                 {
//                     Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
//                     Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
//                     Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
//                     Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
//                 }

//                 int loc_offset_A = (n_hidden_layers + 1) * M * WIDTH + total_offset_A;
//                 // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block0, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 0 * SG_SIZE, WIDTH);
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block1, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 1 * SG_SIZE, WIDTH);
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block2, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 2 * SG_SIZE, WIDTH);
//                 sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                     sg, A_block3, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 3 * SG_SIZE, WIDTH);

//                 sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A)),
//                             sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A]))));
//                 sg.store<8>(global_ptr<int32_t>((int32_t *)(intermediate_output_forward + loc_offset_A + 4 * WIDTH)),
//                             sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A + 4 * WIDTH]))));

//                 /// Compute L2 loss and gradients as input for backward pass

//                 const float inv_N_total_elements = 2.0f / (M * WIDTH);
//                 for (int elemiter = 0; elemiter < TM * WIDTH / 2; elemiter += SG_SIZE) { // row
//                     const int32_t tmp_target =
//                         sg.load((int32_t *)(targets_ptr + total_offset_A) + elemiter); // hbm access. May be slow
//                     int32_t tmp_source = sg.load((int32_t *)&Atmp[sg_offset_A + 2 * elemiter]);
//                     ((bf16 *)&tmp_source)[0] -= ((bf16 *)&tmp_target)[0];
//                     ((bf16 *)&tmp_source)[1] -= ((bf16 *)&tmp_target)[1];
//                     ((bf16 *)&tmp_source)[0] *= inv_N_total_elements;
//                     ((bf16 *)&tmp_source)[1] *= inv_N_total_elements;
//                     sg.store(((int32_t *)&Atmp[sg_offset_A + 2 * elemiter]), tmp_source);
//                 }

//                 // A tmp now holds the grads. We can start the backward pass.

//                 /// ATTENTION: this version only works for K = SGS_IN_WG and NBLOCKCOLS_PER_SG = 4
//                 item.barrier(sycl::access::fence_space::local_space);
//                 sg.store<2>(local_ptr<int32_t>((int32_t *)(&B[sg_offset_B])),
//                             sg.load<2>(global_ptr<int32_t>(
//                                 (int32_t *)(weightsT_ptr + n_hidden_layers * WIDTH * WIDTH + sg_offset_B))));

//                 // we do not need SLM barrier since each SG writes and reads only its own data.
//                 // load A matrix from slm to joint_matrices. This is done to avoid inefficient bf16 HBM access
//                 joint_matrix_load(sg, A_block0, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 0 * SG_SIZE, WIDTH);
//                 joint_matrix_load(sg, A_block1, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 1 * SG_SIZE, WIDTH);
//                 joint_matrix_load(sg, A_block2, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 2 * SG_SIZE, WIDTH);
//                 joint_matrix_load(sg, A_block3, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 3 * SG_SIZE, WIDTH);

//                 // activate the A_blocks with output activation based on forwar
//                 tmp16avalues0 = sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A])));
//                 tmp16avalues1 = sg.load<8>(local_ptr<int32_t>((int32_t *)(&Atmp[sg_offset_A + 4 * WIDTH])));

//                 loc_offset_A = n_hidden_layers * M * WIDTH + total_offset_A;

//                 /// store the activated a values of the input to intermediate output
//                 for (uint8_t iter = 0; iter < TM; iter++) {
//                     *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id) =
//                         tmp16avalues0[iter];
//                     *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id + SG_SIZE) =
//                         tmp16avalues1[iter];
//                 }

//                 // We have n_hidden_layers. Thus n_hidden_layers - 1 gemms between
//                 // the layers (layer 0 -> GEMM -> layer1 -> GEMM -> layer2 -> etc.)
//                 // Since we also do the GEMM from input to hidden layer 0,
//                 // we perform n_hidden_layers GEMMS.
//                 for (uint8_t layer = n_hidden_layers; layer > 0; layer--) // we are also doing output->last hidden
//                 layer
//                 {

//                     joint_matrix_fill(sg, C_block0, 0.0f);
//                     joint_matrix_fill(sg, C_block1, 0.0f);
//                     joint_matrix_fill(sg, C_block2, 0.0f);
//                     joint_matrix_fill(sg, C_block3, 0.0f);

//                     item.barrier(sycl::access::fence_space::local_space); // wait for B to be done storing

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block0, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block0, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block0, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 0 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block0, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block1, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block1, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block1, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 1 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block1, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block2, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block2, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block2, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 2 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block2, B_block, C_block3);

//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 0 * 2 * TN, 2 * WIDTH);
//                     C_block0 = joint_matrix_mad(sg, A_block3, B_block, C_block0);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 1 * 2 * TN, 2 * WIDTH);
//                     C_block1 = joint_matrix_mad(sg, A_block3, B_block, C_block1);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 2 * 2 * TN, 2 * WIDTH);
//                     C_block2 = joint_matrix_mad(sg, A_block3, B_block, C_block2);
//                     joint_matrix_load(sg, B_block, local_ptr<bf16>(&B[0]) + 3 * WIDTH * TK + 3 * 2 * TN, 2 * WIDTH);
//                     C_block3 = joint_matrix_mad(sg, A_block3, B_block, C_block3);

//                     // load B for next iteration into SLM
//                     if (layer > 1) {
//                         item.barrier(sycl::access::fence_space::local_space);
//                         // ((int32_t *)(&B[sg_offset_B]))[loc_id] = ((int32_t*)(weightsT_ptr+(layer-1)*WIDTH*WIDTH +
//                         // sg_offset_B))[loc_id];
//                         // ((int32_t *)(&B[sg_offset_B]))[loc_id+SG_SIZE] =
//                         // ((int32_t*)(weightsT_ptr+(layer-1)*WIDTH*WIDTH + sg_offset_B))[loc_id+SG_SIZE];
//                         sg.store<2>(local_ptr<int32_t>((int32_t *)(&B[sg_offset_B])),
//                                     sg.load<2>(global_ptr<int32_t>(
//                                         (int32_t *)(weightsT_ptr + (layer - 1) * WIDTH * WIDTH + sg_offset_B))));
//                     }

//                     auto Ci_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block0);
//                     auto Ai_data0 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block0);
//                     auto Ci_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block1);
//                     auto Ai_data1 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block1);
//                     auto Ci_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block2);
//                     auto Ai_data2 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block2);
//                     auto Ci_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, C_block3);
//                     auto Ai_data3 = sycl::ext::intel::experimental::matrix::get_wi_data(sg, A_block3);

//                     for (uint8_t rowiter = 0; rowiter < Ci_data0.length(); rowiter++) // should be TM in length
//                     {
//                         Ai_data0[rowiter] = (bf16)Ci_data0[rowiter];
//                         Ai_data1[rowiter] = (bf16)Ci_data1[rowiter];
//                         Ai_data2[rowiter] = (bf16)Ci_data2[rowiter];
//                         Ai_data3[rowiter] = (bf16)Ci_data3[rowiter];
//                     }

//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block0, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 0 * SG_SIZE, WIDTH);
//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block1, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 1 * SG_SIZE, WIDTH);
//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block2, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 2 * SG_SIZE, WIDTH);
//                     sycl::ext::intel::experimental::matrix::joint_matrix_store(
//                         sg, A_block3, local_ptr<bf16>(&Atmp[0]) + sg_offset_A + 3 * SG_SIZE, WIDTH);

//                     const int loc_offset_A = (layer - 1) * M * WIDTH + total_offset_A;
//                     for (uint8_t iter = 0; iter < TM; iter++) {
//                         *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id) =
//                             *(((int32_t *)&Atmp[sg_offset_A + iter * WIDTH]) + loc_id);

//                         *((int32_t *)(intermediate_output_backward + loc_offset_A + iter * WIDTH) + loc_id + SG_SIZE)
//                         =
//                             *(((int32_t *)&Atmp[sg_offset_A + iter * WIDTH]) + loc_id + SG_SIZE);
//                     }
//                 }
//             });
//     });

//     std::vector<sycl::event> events(n_hidden_layers + 1);
//     for (int iter = 0; iter < n_hidden_layers + 1; iter++) {
//         events[iter] = oneapi::mkl::blas::row_major::gemm(
//             q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, WIDTH, WIDTH, M, 1.0f,
//             reinterpret_cast<const oneapi::mkl::bfloat16 *>(intermediate_output_forward) + iter * M * WIDTH, WIDTH,
//             reinterpret_cast<oneapi::mkl::bfloat16 *>(intermediate_output_backward) + iter * M * WIDTH, WIDTH, 1.0f,
//             reinterpret_cast<oneapi::mkl::bfloat16 *>(output_ptr) + iter * WIDTH * WIDTH, WIDTH, {e});
//     }

//     return events;
// }

} // namespace kernels
} // namespace tinydpcppnn
