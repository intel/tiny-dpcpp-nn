/**
 * @file gemm.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implements gemms for the backward pass
 * @version 0.1
 * @date 2024-005-27
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */


#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;


namespace sycl::ext::intel::esimd::xmx {
    template <int SystolicDepth, int RepeatCount, typename T, typename CT, typename BT, typename AT,
          dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
          dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(), int N, int N_orig, int BN, int AN,
          int BN_orig>
    __ESIMD_NS::simd<T, N> dpas(__ESIMD_NS::simd_view<simd<CT, N_orig>, region1d_t<CT, N, 1>> C, __ESIMD_NS::simd_view<simd<BT, BN_orig>, region1d_t<BT, BN, 1>> B, __ESIMD_NS::simd<AT, AN> A) {
        (void)detail::verify_parameters_and_deduce_exec_size<SystolicDepth, RepeatCount, T, CT, BT, AT, BPrecision,
                                                            APrecision, BN, AN>();

        using MsgT = int;
        constexpr int ANCasted = AN * sizeof(AT) / sizeof(MsgT);
        constexpr int BNCasted = BN * sizeof(BT) / sizeof(MsgT);
        __ESIMD_NS::simd<MsgT, ANCasted> ACasted = A.template bit_cast_view<MsgT>();
        __ESIMD_NS::simd<MsgT, BNCasted> BCasted = B.template bit_cast_view<MsgT>();
        using CRawT = typename __ESIMD_NS::simd<CT, N>::raw_element_type;
        using RawT = typename __ESIMD_NS::simd<T, N>::raw_element_type;
        return __esimd_dpas2<BPrecision, APrecision, SystolicDepth, RepeatCount, RawT, CRawT, MsgT, MsgT, N, BNCasted,
                            ANCasted>(C.data(), BCasted.data(), ACasted.data());
    }
    }; // namespace sycl::ext::intel::esimd::xmx


template <typename T, int WIDTH>
class Gemm 
{
public:
    /** Performs a batched Gemm of sizes WIDTHxM, MxWIDTH = WIDTHxWIDTH
    * and nbatches number of batches. We assume that the A matrix is col major
    * the B matrix is row major and the output is row-major
    *
    */
    static void batched(const size_t M, const int nbatches, const T * const A, const T * const B, T * const C, sycl::queue &q)
    {
        static_assert(WIDTH % 16 == 0, "WIDTH must be a multiple of 16");
        static_assert(std::is_same<T, sycl::half>::value || std::is_same<T, bf16>::value, "T must be fp16 or bf16");

        constexpr int TN = 16;
        constexpr int TK = 8 * std::min<int>(8, 32 / (8 * sizeof(T)));
        assert(M%TK == 0);

        //one sg per row now
        q.parallel_for(sycl::nd_range<3>(sycl::range<3>(nbatches, WIDTH, 1), sycl::range<3>(1,1,1)), [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL 
        {
                const int matrix = item.get_global_id(0);
                const int row = item.get_global_id(1);                

                T const * A_loc = A + matrix * M * WIDTH + row; //col-major
                T const * B_loc = B + matrix * M*WIDTH; //row-major
                T * C_loc = C + matrix * WIDTH * WIDTH + row * WIDTH; //row-major

                simd<float, WIDTH> blockC = 0.0f; //the whole row of this item
                
                simd<int,TK> offsets_A(0, sizeof(T)*WIDTH);
                for (int inneriter = 0; inneriter < M; inneriter+=TK) {
                    simd<T, TK> blockA = lsc_gather(A_loc + inneriter * WIDTH, offsets_A);
                    simd<T, TK*WIDTH> blockB = LoadB<TK, TN>(B_loc + inneriter * WIDTH);
                    for (int chunkiter = 0; chunkiter < WIDTH; chunkiter+=TN) {
                        blockC.template select<TN, 1>(chunkiter) = xmx::dpas<8, 1, float>(
                            blockC.template select<TN, 1>(chunkiter), blockB.template select<TK*TN, 1>(chunkiter*TK), blockA);
                    }
                }

                lsc_block_store(C_loc, convert<T, float>(blockC));
            }).wait();
    }

private: 
    /**
    * Load a chunk of B sized TKxWIDTH in blocks of TK*TN and vnni it.
    */
    template <int TK, int TN>
    static simd<T, TK*WIDTH> LoadB(T const * const B) {
        static_assert(sizeof(T) == 2);
        static_assert(WIDTH%TN == 0);

        simd<T, TK*WIDTH> output;
        for (int coliter = 0; coliter < WIDTH; coliter += TN) {
            for (int rowiter = 0; rowiter < TK; rowiter+=2) {
                output.template select<TN, 2>(coliter*TK + rowiter*TN) = lsc_block_load<T, TN>(B + coliter + rowiter*WIDTH);
                output.template select<TN, 2>(coliter*TK + rowiter*TN + 1) = lsc_block_load<T, TN>(B + coliter + (rowiter+1)*WIDTH);
            }
        }

        return output;
    }

};

