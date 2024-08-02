/**
 * @file kernel_esimd.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Esimd implementation of the forward, backward and inference kernels class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <algorithm>
#include <optional>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "DeviceMatrix.h"
#include "common.h"
#include "oneapi/mkl.hpp"

namespace sycl::ext::intel::esimd::xmx {
template <int SystolicDepth, int RepeatCount, typename T, typename CT, typename BT, typename AT,
          dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
          dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(), int N, int N_orig, int BN, int AN,
          int AN_orig>
__ESIMD_NS::simd<T, N> dpas(__ESIMD_NS::simd_view<simd<CT, N_orig>, region1d_t<CT, N, 1>> C, __ESIMD_NS::simd<BT, BN> B,
                            __ESIMD_NS::simd_view<simd<AT, AN_orig>, region1d_t<AT, AN, 1>> A) {
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

template <int SystolicDepth, int RepeatCount, typename T, typename BT, typename AT,
          dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
          dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(), int BN, int AN, int AN_orig>
auto dpas(__ESIMD_NS::simd<BT, BN> B, __ESIMD_NS::simd_view<simd<AT, AN_orig>, region1d_t<AT, AN, 1>> A) {

    constexpr int ExecutionSize = detail::verify_parameters_and_deduce_exec_size<SystolicDepth, RepeatCount, T, T, BT,
                                                                                 AT, BPrecision, APrecision, BN, AN>();
    // Result(_Mx_N) = A(_Mx_K) * B(_Kx_N)
    // where:
    //   _M = RepeatCount;
    //   _K = SystolicDepth * OpsPerChannel;
    //   _N = ExecutionSize (unknown, but deducible), must be 8 or 16.
    constexpr int ResultN = RepeatCount * ExecutionSize;

    using MsgT = int;
    constexpr int ANCasted = AN * sizeof(AT) / sizeof(MsgT);
    constexpr int BNCasted = BN * sizeof(BT) / sizeof(MsgT);
    __ESIMD_NS::simd<MsgT, ANCasted> ACasted = A.template bit_cast_view<MsgT>();
    __ESIMD_NS::simd<MsgT, BNCasted> BCasted = B.template bit_cast_view<MsgT>();

    constexpr int Info = (RepeatCount << 24) + (SystolicDepth << 16) + ((int)APrecision << 8) + (int)BPrecision;
    using RawT = typename __ESIMD_NS::simd<T, ResultN>::raw_element_type;
    __ESIMD_NS::simd<T, ResultN> Result =
        __esimd_dpas_nosrc0<Info, RawT, MsgT, MsgT, ResultN, BNCasted, ANCasted>(BCasted.data(), ACasted.data());
    return Result;
}

}; // namespace sycl::ext::intel::esimd::xmx

namespace tinydpcppnn {
namespace kernels {
namespace esimd {

using namespace sycl::ext::intel::esimd;
using sycl::ext::intel::experimental::esimd::cache_hint;
using namespace sycl::ext::intel::experimental::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;

/**
 * @brief Struct to decide type for accumulation (CType) for XMX at compile time,
 * depending on the given type T.
 *
 * Currently returns CType == T, except when T == bf16, then CType == float.
 *
 * @tparam T
 */
template <typename T> struct XMXCType {
    typedef T CType;
};
template <> struct XMXCType<bf16> {
    typedef float CType;
};
template <> struct XMXCType<sycl::half> {
#if TARGET_DEVICE == 0
    typedef float CType;
    // typedef sycl::half CType;
#elif TARGET_DEVICE == 1
    typedef float CType;
#endif
};

/**
 * @brief Struct which gives us the value to use for TN in the dpas instruction
 * Depending on the device.
 *
 */
struct XMXTn {
#if TARGET_DEVICE == 0
    static constexpr int TN = 16;
#elif TARGET_DEVICE == 1
    static constexpr int TN = 8;
#endif
};

/**
 * @brief Struct to give us the maximum number of bytes in a send instruction,
 * depending on the device
 *
 */
struct XMXMaxSendBytes {
#if TARGET_DEVICE == 0
    static constexpr int MaxBytes = 512;
#elif TARGET_DEVICE == 1
    static constexpr int MaxBytes = 256;
#endif
};

/**
 * @brief
 *
 * @tparam T type for the computations. Everything that is supported by xmx::dpas
 * should work fine.
 * @tparam INPUT_WIDTH The width of the input layer of the network. In general
 * it should be a multiple of TK, right now it is equal to WIDTH.
 * @tparam WIDTH Denotes the width of every hidden layer, may be 16, 32, 64, 128.
 * @tparam OUTPUT_WIDTH The width of the output layer, currently equal to WIDTH. Later a multiple of TN.
 * @tparam activation Activation function. Currently either none or ReLU.
 * @tparam output_activation Activation for the output layer. Currently None.
 * @tparam TN Device dependent, whatever is supported by the chosen device. 8 for DG2, 16 for PVC.
 */
template <typename T, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation, Activation output_activation>
class EsimdKernels {

    using Tc = typename XMXCType<T>::CType;
    static constexpr int TN = XMXTn::TN;

  public:
    static std::vector<sycl::event> forward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                 const DeviceMatrixView<T> &input,
                                                 DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                                                 const std::vector<sycl::event> &deps) {
        return forward_impl_general<false>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    static std::vector<sycl::event> backward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                  const DeviceMatrixView<T> &input, DeviceMatricesView<T> output,
                                                  DeviceMatricesView<T> intermediate_backward,
                                                  const DeviceMatricesView<T> &intermediate_forward,
                                                  const int n_hidden_layers, const std::vector<sycl::event> &deps,
                                                  std::optional<DeviceMatrixView<T>> dL_dinput = std::nullopt) {
        // make sure there is no remainder and no out of bounds accesses
        static_assert(WIDTH % TN == 0);
        // only works for input_width == width == output_width
        static_assert(INPUT_WIDTH == WIDTH);
        static_assert(OUTPUT_WIDTH == WIDTH);
        const size_t M = input.m();

        constexpr int TM = ComputeTM();

        assert(M % TM == 0);
        const int ITEMS_IN_WG = ComputeItemsInWG(M, TM);
        constexpr int TK = ComputeTK();
        static_assert(WIDTH % TK == 0);

        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);
            sycl::ext::oneapi::experimental::properties properties{
                sycl::ext::intel::experimental::fp_control<sycl::ext::intel::experimental::fp_mode::denorm_hf_allow>};
            cgh.parallel_for(
                sycl::nd_range<1>(M / TM, ITEMS_IN_WG), properties, [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    const size_t loc_row_offset = item.get_global_linear_id() * TM;

                    simd<T, TM * WIDTH> As;
                    loadRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(input.GetPointer(loc_row_offset, 0),
                                                                                As); // block-major with TMxTK blocks

                    // store backward activated input to the last intermediate output
                    if constexpr (output_activation == Activation::None) {
                        // do nothing
                    } else {
                        // Compute derivative of activation function
                        applyBackwardActivation<output_activation, TM, TK, TK>(
                            intermediate_forward.GetElementPointer(n_hidden_layers + 1, loc_row_offset, 0), As, As);
                        // n_hidden_layers +1 because the first (0 index) matrix in intermediate_forward (view of
                        // DeviceMatrices object) is input
                    }

                    // store activated in intermediate output
                    storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                        As, intermediate_backward.GetElementPointer(n_hidden_layers, loc_row_offset, 0));
                    simd<Tc, TM * WIDTH> Cs;
                    // we are also doing output->last hidden layer

                    for (int layer = n_hidden_layers; layer > 0; layer--) {
                        MAD<TM, TK>(As, weights.GetMatrixPointer(layer), Cs);

                        applyBackwardActivation<activation, TM, TK, TN>(
                            intermediate_forward.GetElementPointer(layer, loc_row_offset, 0), Cs, As);

                        storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                            As, intermediate_backward.GetElementPointer(layer - 1, loc_row_offset, 0));
                    }
                    if (dL_dinput.has_value()) {
                        MAD<TM, TK>(As, weights.GetMatrixPointer(0), Cs);

                        applyBackwardActivation<Activation::None, TM, TK, TN>(
                            intermediate_forward.GetElementPointer(0, loc_row_offset, 0), Cs, As);

                        storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                            As, dL_dinput->GetPointer(loc_row_offset, 0));
                    }
                });
        });

        const auto a = intermediate_backward.GetMatrixPointer(0);
        const auto b = intermediate_forward.GetMatrixPointer(0);
        const auto c = output.GetMatrixPointer(0);

        return {oneapi::mkl::blas::row_major::gemm_batch(
            q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, static_cast<int64_t>(WIDTH),
            static_cast<int64_t>(WIDTH), static_cast<int64_t>(M), 1.0f, a, static_cast<int64_t>(WIDTH),
            static_cast<int64_t>(WIDTH * M), b, static_cast<int64_t>(WIDTH), static_cast<int64_t>(WIDTH * M), 0.0f, c,
            static_cast<int64_t>(WIDTH), static_cast<int64_t>(WIDTH * WIDTH), n_hidden_layers + 1,
            oneapi::mkl::blas::compute_mode::unset, {e})};
    }

    static std::vector<sycl::event> inference_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                   const DeviceMatrixView<T> &input,
                                                   DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                                                   const std::vector<sycl::event> &deps) {
        return forward_impl_general<true>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    /*************the following functions are only public for testing purposes*******************/

    // in register everything is in block major format with blocks of size TMxTK
    template <int TM, int TK, cache_hint L1, cache_hint L3, int TMWIDTH>
    SYCL_ESIMD_FUNCTION static void storeRow(simd<T, TMWIDTH> &src, T *const dest) {

        static_assert(TM == 1 || TM == 2 || TM == 4 || TM == 8);
        static_assert(WIDTH % TK == 0);
        static_assert(TMWIDTH == TM * WIDTH);
        static_assert(sizeof(T) <= 4);

        constexpr int rows_per_load = std::min<int>(XMXMaxSendBytes::MaxBytes / (WIDTH * sizeof(T)), TM);
        static_assert(rows_per_load > 0);
        auto src_2d = src.template bit_cast_view<T, TMWIDTH / TK, TK>(); // block major

#pragma unroll
        for (int row = 0; row < TM; row += rows_per_load) {
            simd<T, WIDTH * rows_per_load> tmp;
#pragma unroll
            for (int locrowiter = 0; locrowiter < rows_per_load; locrowiter++) {
                tmp.template select<WIDTH, 1>(locrowiter * WIDTH) =
                    src_2d.template select<WIDTH / TK, TM, TK, 1>(row + locrowiter, 0);
            }
            lsc_block_store<T, rows_per_load * WIDTH, lsc_data_size::default_size, L1, L3>(dest + row * WIDTH, tmp,
                                                                                           overaligned_tag<8>());
        }
    }

    // in register everything is in block major format with blocks of size TMxTK
    template <int TM, int TK, cache_hint L1, cache_hint L3, int TMWIDTH>
    SYCL_ESIMD_FUNCTION static void loadRow(T const *const src, simd<T, TMWIDTH> &dest) {
        static_assert(TM == 1 || TM == 2 || TM == 4 || TM == 8);
        static_assert(WIDTH % TK == 0);
        static_assert(TMWIDTH == TM * WIDTH);
        static_assert(sizeof(T) <= 4);

// DG2 does not have 2d send instructions
#if TARGET_DEVICE == 0
        constexpr int elems_per_pos = 4 / sizeof(T);
        constexpr int blocks_per_load = TK * elems_per_pos > WIDTH ? 1 : elems_per_pos;
        constexpr int nloads = WIDTH / (TK * blocks_per_load);
        static_assert(nloads > 0);
        auto dest_int = dest.template bit_cast_view<int32_t>();
#pragma unroll
        for (int load_iter = 0; load_iter < nloads; load_iter++) {
            dest_int.template select<TM * TK / elems_per_pos * blocks_per_load, 1>(TM * TK / elems_per_pos *
                                                                                   blocks_per_load * load_iter) =
                lsc_load_2d<int32_t, TK / elems_per_pos, TM, blocks_per_load, false, false, L1, L3>(
                    reinterpret_cast<int32_t const *>(src), WIDTH * sizeof(T) - 1, TM - 1, WIDTH * sizeof(T) - 1,
                    load_iter * TK, 0);
        }
#elif TARGET_DEVICE == 1
        constexpr int elems_per_load = std::min<int>(TMWIDTH, XMXMaxSendBytes::MaxBytes / sizeof(T));
        constexpr int rows_per_load = elems_per_load / WIDTH;
        static_assert(rows_per_load > 0 && TM % rows_per_load == 0);
        static_assert(elems_per_load % WIDTH == 0);
        static_assert(TMWIDTH % elems_per_load == 0);

        for (int loaditer = 0; loaditer < TM; loaditer += rows_per_load) {

            simd<T, elems_per_load> tmp =
                lsc_block_load<T, elems_per_load, lsc_data_size::default_size, L1, L3>(src + loaditer * WIDTH);
#pragma collapse(2) unroll
            for (int blockcoliter = 0; blockcoliter < WIDTH; blockcoliter += TK) {
                for (int rowiter = 0; rowiter < rows_per_load; rowiter++) {
                    dest.template select<TK, 1>(loaditer * TK + blockcoliter * TM + rowiter * TK) =
                        tmp.template select<TK, 1>(blockcoliter + rowiter * WIDTH);
                }
            }
        }
#endif
    }

    // we are assuming a block major layout and vnni'd B
    template <int TM, int TK, int TMWIDTH>
    SYCL_ESIMD_FUNCTION static void MAD(simd<T, TMWIDTH> &As, T const *const __restrict__ B, simd<Tc, TMWIDTH> &Cs) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(TMWIDTH % TM == 0);
        static_assert(TMWIDTH / TM == WIDTH);
        static_assert(WIDTH % TK == 0 && WIDTH % TN == 0);
        static_assert(sizeof(T) <= 4 && sizeof(Tc) <= 4);
        constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(T));

#if TARGET_DEVICE == 0
        std::array<config_2d_mem_access<float, TN, TK / vnni_factor, 1>, WIDTH / TN> configs;
        for (int iterB = 0; iterB < WIDTH / TN; iterB++) {
            configs[iterB] = config_2d_mem_access<float, TN, TK / vnni_factor, 1>(
                reinterpret_cast<float const *>(B), vnni_factor * WIDTH * sizeof(T) - 1, WIDTH / vnni_factor - 1,
                vnni_factor * WIDTH * sizeof(T) - 1, iterB * TN, 0);
        }

#pragma unroll
        for (int iterB = 0; iterB < WIDTH; iterB += TN) {
            simd<T, TK * TN> BlockB;
            auto BlockB_float = BlockB.template bit_cast_view<float>();
            BlockB_float = my_2d_load<float, TK / vnni_factor, TN>(configs[iterB / TN]);
            // lsc_load_2d<float, TN, TK / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
            //     configs[iterB / TN]);
            Cs.template select<TM * TN, 1>(iterB * TM) =
                xmx::dpas<8, TM, Tc>(BlockB, As.template select<TM * TK, 1>(0));
        }

#pragma unroll
        for (int iterA = TK; iterA < WIDTH; iterA += TK) {
            for (int iterB = 0; iterB < WIDTH / TN; iterB++) {
                configs[iterB].set_y(iterA / vnni_factor);
            }

#pragma unroll
            for (int iterB = 0; iterB < WIDTH; iterB += TN) {
                simd<T, TK * TN> BlockB;
                // config.set_x(iterB);
                auto BlockB_float = BlockB.template bit_cast_view<float>();
                BlockB_float = my_2d_load<float, TK / vnni_factor, TN>(configs[iterB / TN]);
                // lsc_load_2d<float, TN, TK / vnni_factor, 1, false, false, cache_hint::cached, cache_hint::cached>(
                //     configs[iterB / TN]);

                Cs.template select<TM * TN, 1>(iterB * TM) = xmx::dpas<8, TM, Tc>(
                    Cs.template select<TM * TN, 1>(iterB * TM), BlockB, As.template select<TM * TK, 1>(iterA * TM));
            }
        }
#elif TARGET_DEVICE == 1
        static_assert(TN == 8);
        static_assert(WIDTH >= 16); // TODO: generalize
        static_assert(WIDTH % (2 * TN) == 0);
        // As TN == 8, even vnni'ed we would only use half the cache line using a single block.
        // Thus, we load 2 blocks or more at the same time.
        Cs = static_cast<Tc>(0);
        if constexpr (WIDTH >= 4 * TN) {
            static_assert(WIDTH % (4 * TN) == 0);
            for (int iterA = 0; iterA < WIDTH; iterA += TK) {
                auto current_A = As.template select<TM * TK, 1>(iterA * TM);
                // #pragma unroll(2)
                for (int iterB = 0; iterB < WIDTH; iterB += 4 * TN) {
                    simd<T, TK * TN> BlockB0;
                    simd<T, TK * TN> BlockB1;
                    simd<T, TK * TN> BlockB2;
                    simd<T, TK * TN> BlockB3;
                    auto BlockB0_float = BlockB0.template bit_cast_view<float>();
                    auto BlockB1_float = BlockB1.template bit_cast_view<float>();
                    auto BlockB2_float = BlockB2.template bit_cast_view<float>();
                    auto BlockB3_float = BlockB3.template bit_cast_view<float>();

                    for (int rowiter = 0; rowiter < TK / vnni_factor; rowiter++) {
                        auto tmp_reg = lsc_block_load<float, 4 * TN, lsc_data_size::default_size, cache_hint::cached,
                                                      cache_hint::cached>(
                            reinterpret_cast<float const *>(B) + iterB + iterA / vnni_factor * WIDTH + rowiter * WIDTH);
                        BlockB0_float.template select<TN, 1>(rowiter * TN) = tmp_reg.template select<TN, 1>(0);
                        BlockB1_float.template select<TN, 1>(rowiter * TN) = tmp_reg.template select<TN, 1>(TN);
                        BlockB2_float.template select<TN, 1>(rowiter * TN) = tmp_reg.template select<TN, 1>(2 * TN);
                        BlockB3_float.template select<TN, 1>(rowiter * TN) = tmp_reg.template select<TN, 1>(3 * TN);
                    }

                    Cs.template select<TM * TN, 1>(iterB * TM) =
                        xmx::dpas<8, TM, Tc>(Cs.template select<TM * TN, 1>(iterB * TM), BlockB0, current_A);
                    Cs.template select<TM * TN, 1>((iterB + TN) * TM) =
                        xmx::dpas<8, TM, Tc>(Cs.template select<TM * TN, 1>((iterB + TN) * TM), BlockB1, current_A);
                    Cs.template select<TM * TN, 1>((iterB + 2 * TN) * TM) =
                        xmx::dpas<8, TM, Tc>(Cs.template select<TM * TN, 1>((iterB + 2 * TN) * TM), BlockB2, current_A);
                    Cs.template select<TM * TN, 1>((iterB + 3 * TN) * TM) =
                        xmx::dpas<8, TM, Tc>(Cs.template select<TM * TN, 1>((iterB + 3 * TN) * TM), BlockB3, current_A);
                }
            }
        } else if constexpr (WIDTH == 2 * TN) {
            for (int iterA = 0; iterA < WIDTH; iterA += TK) {
                // #pragma unroll
                //                 for (int iterB = 0; iterB < WIDTH; iterB += 2 * TN) {
                simd<T, TK * TN> BlockB0;
                simd<T, TK * TN> BlockB1;
                auto BlockB0_float = BlockB0.template bit_cast_view<float>();
                auto BlockB1_float = BlockB1.template bit_cast_view<float>();

#pragma unroll
                for (int rowiter = 0; rowiter < TK / vnni_factor; rowiter++) {
                    auto tmp_reg = lsc_block_load<float, 2 * TN, lsc_data_size::default_size, cache_hint::cached,
                                                  cache_hint::cached>(reinterpret_cast<float const *>(B) + /*iterB +*/
                                                                      iterA / vnni_factor * WIDTH + rowiter * WIDTH);
                    BlockB0_float.template select<TN, 1>(rowiter * TN) = tmp_reg.template select<TN, 1>(0);
                    BlockB1_float.template select<TN, 1>(rowiter * TN) = tmp_reg.template select<TN, 1>(TN);
                }

                Cs.template select<TM * TN, 1>(/*iterB * TM*/ 0) =
                    xmx::dpas<8, TM, Tc>(Cs.template select<TM * TN, 1>(/*iterB * TM*/ 0), BlockB0,
                                         As.template select<TM * TK, 1>(iterA * TM));
                Cs.template select<TM * TN, 1>((/*iterB + */ TN) * TM) =
                    xmx::dpas<8, TM, Tc>(Cs.template select<TM * TN, 1>((/*iterB + */ TN) * TM), BlockB1,
                                         As.template select<TM * TK, 1>(iterA * TM));
                //}
            }
        }
#endif
    }

    template <Activation act, int TM, int TK, int N, typename Tsrc, typename Tdest>
    SYCL_ESIMD_FUNCTION static void applyActivation(simd<Tsrc, N> &Src, simd<Tdest, N> &Dest) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(TK == 8 || TK == 16 || TK == 32 || TK == 64);

        if constexpr (act == Activation::None) {
            reBlock<TM, TK, TN>(convert<Tdest, Tsrc>(Src), Dest);
        } else if constexpr (act == Activation::ReLU) {
            reBlock<TM, TK, TN>(max<Tdest>(convert<Tdest, Tsrc>(Src), simd<Tdest, N>(static_cast<Tdest>(0))), Dest);
        } else if constexpr (act == Activation::Sigmoid) {
            // Convert bfloat16 vectors to float to perform arithmetic operations.
            simd<float, N> sigmoid_result_float = 1.0f / (1.0f + esimd::exp(-convert<float>(Src)));
            reBlock<TM, TK, TN>(convert<Tdest>(sigmoid_result_float), Dest);
        }
    }

    template <Activation act, int TM, int COLS_OUT, int COLS_IN, int N, typename Tdec, typename Tsrc, typename Tdest>
    SYCL_ESIMD_FUNCTION static void applyBackwardActivation(Tdec const *const Dec, simd<Tsrc, N> &Src,
                                                            simd<Tdest, N> &Dest) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(COLS_OUT == 8 || COLS_OUT == 16 || COLS_OUT == 32 || COLS_OUT == 64);
        static_assert(COLS_IN == 8 || COLS_IN == 16 || COLS_IN == 32 || COLS_IN == 64);
        static_assert(N == TM * WIDTH);
        static_assert(WIDTH % COLS_OUT == 0);
        static_assert(WIDTH % COLS_IN == 0);

        if constexpr (act == Activation::None) {
            reBlock<TM, COLS_OUT, COLS_IN>(convert<Tdest, Tsrc>(Src), Dest);
        } else if constexpr (act == Activation::ReLU) {
            simd<Tdec, N> loc_dec;
            loadRow<TM, COLS_IN, cache_hint::uncached, cache_hint::uncached>(Dec, loc_dec);
            simd_mask<N> m = loc_dec <= simd<Tdec, N>(0);
            Src.merge(simd<Tsrc, N>(0), m); // ATTENTION: this changes Src.
            reBlock<TM, COLS_OUT, COLS_IN>(convert<Tdest, Tsrc>(Src), Dest);
        } else if constexpr (act == Activation::Sigmoid) {
            // The derivative of the sigmoid is sigmoid(x) * (1 - sigmoid(x))
            simd<Tdec, N> loc_dec;
            loadRow<TM, COLS_IN, cache_hint::uncached, cache_hint::uncached>(Dec, loc_dec);
            simd<float, N> sigmoid_result = 1.0f / (1.0f + esimd::exp(-convert<float>(loc_dec)));
            simd<float, N> sigmoid_derivative = sigmoid_result * (1.0f - sigmoid_result);
            simd<float, N> Src_with_derivative = convert<float>(Src) * sigmoid_derivative;
            reBlock<TM, COLS_OUT, COLS_IN>(convert<Tdest>(Src_with_derivative), Dest);
        }
    }

    // TK == 8, 16, 32, 64; TN == 8 (DG2), 16 (PVC)
    // Src contains of blocks sized TM*locTN
    // Dest of blocks sized TM*TK
    template <int TM, int TK, int locTN, int N>
    SYCL_ESIMD_FUNCTION static void reBlock(simd<T, N> Src, simd<T, N> &Dest) {
        static_assert(TK == locTN || TK == 2 * locTN || TK == 4 * locTN || TK == 8 * locTN || TK == locTN / 2);

        if constexpr (TK == locTN) {
            Dest = Src;
        } else if constexpr (TK == 2 * locTN || TK == 4 * locTN || TK == 8 * locTN) {
            constexpr int FAC = TK / locTN;
            auto Dest_2d = Dest.template bit_cast_view<T, N / TK, TK>(); // block major 2d layout
            for (int iter = 0; iter < WIDTH / locTN; iter++) {           // iterate over the blocks in SRC
                Dest_2d.template select<TM, 1, locTN, 1>((iter / FAC) * TM, (iter % FAC) * locTN) =
                    Src.template select<TM * locTN, 1>(iter * TM * locTN);
            }
        } else if constexpr (TK == locTN / 2) {
            auto Src_2d = Src.template bit_cast_view<T, N / locTN, locTN>(); // block major 2d layout
            for (int iter = 0; iter < WIDTH / TK; iter++) {                  // iterate over the blocks in SRC
                Dest.template select<TM * TK, 1>(iter * TM * TK) =
                    Src_2d.template select<TM, 1, TK, 1>((iter / 2) * TM, (iter % 2) * TK);
            }
        }
    }

  private:
    static constexpr int ComputeTM() {
#if TARGET_DEVICE == 0
        return 8;
#elif TARGET_DEVICE == 1
        if constexpr (WIDTH < 64)
            return 8;
        else if constexpr (WIDTH >= 64) {
            constexpr int factor = std::max(1, WIDTH / 64); // shut up div by 0 warning
            return std::max<int>(1, 4 / factor);
        }
#endif
    }

    static constexpr int ComputeTK() { return 8 * std::min<int>(8, 32 / (8 * sizeof(T))); }

    static int ComputeItemsInWG(const size_t M, const int TM) {
// TODO: 64 depends on the device. It is different for non-PVC hardware
#if TARGET_DEVICE == 0
        constexpr int max_items_per_wg = 64;
#elif TARGET_DEVICE == 1
        constexpr int max_items_per_wg = 1;
#endif
        int items_in_wg = std::min<int>(M / TM, max_items_per_wg);
        while (M / TM % items_in_wg != 0) {
            items_in_wg--;
        }
        if (items_in_wg <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");

        return items_in_wg;
    }

    template <typename LoadT, int NROWS, int NCOLS>
    static simd<LoadT, NROWS * NCOLS> my_2d_load(config_2d_mem_access<LoadT, NCOLS, NROWS, 1> &acc) {
#if TARGET_DEVICE == 0
        return lsc_load_2d<LoadT, NCOLS, NROWS, 1, false, false, cache_hint::cached, cache_hint::cached>(acc);
#elif TARGET_DEVICE == 1
        simd<LoadT, NROWS * NCOLS> ret;
#pragma unroll
        for (int rowiter = 0; rowiter < NROWS; rowiter++) {
            ret.template select<NCOLS, 1>(rowiter * NCOLS) =
                lsc_block_load<LoadT, NCOLS, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(
                    reinterpret_cast<LoadT const *const>(
                        reinterpret_cast<uint8_t const *const>(acc.get_data_pointer() + acc.get_x()) +
                        ((rowiter + acc.get_y()) * (acc.get_surface_pitch() + 1))));
        }

        return ret;
#endif
    }

    template <bool INFERENCE>
    static std::vector<sycl::event>
    forward_impl_general(sycl::queue &q, const DeviceMatricesView<T> &weights, const DeviceMatrixView<T> &input,
                         DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                         const std::vector<sycl::event> &deps) {

        // throw std::logic_error("General function should not be called.");
        const size_t M = input.m();
        static_assert(INPUT_WIDTH == WIDTH);
        static_assert(OUTPUT_WIDTH == WIDTH);
        static_assert(WIDTH % TN == 0);

        constexpr int TM = ComputeTM();
        // make sure there is no remainder and no out of bounds accesses
        // this may be adjusted in the future
        assert(M % TM == 0);

        // TK depends on the datatype T
        constexpr int TK = ComputeTK();
        const int ITEMS_IN_WG = ComputeItemsInWG(M, TM);

        // One Block Row has TM rows an N columns.
        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);
            sycl::ext::oneapi::experimental::properties properties{
                sycl::ext::intel::experimental::fp_control<sycl::ext::intel::experimental::fp_mode::denorm_hf_allow>};
            cgh.parallel_for(
                sycl::nd_range<1>(M / TM, ITEMS_IN_WG), properties, [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    const size_t loc_row_offset = item.get_global_linear_id() * TM;

                    // we store blocks contiguously
                    simd<T, TM * WIDTH> As;
                    loadRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(input.GetPointer(loc_row_offset, 0),
                                                                                As);

                    // if not inference activate and store in intermediate output
                    if constexpr (!INFERENCE) {
                        storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                            As,
                            intermediate_output.GetElementPointer(0, loc_row_offset, 0)); // saving non-activated input
                    }

                    simd<Tc, TM * WIDTH> Cs;
                    for (int layer = 0; layer < n_hidden_layers; layer++) {
                        // reset result matrices

                        MAD<TM, TK>(As, weights.GetMatrixPointer(layer), Cs);

                        // activate and save
                        applyActivation<activation, TM, TK>(Cs, As);

                        if constexpr (!INFERENCE)
                            storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                                As, intermediate_output.GetElementPointer(
                                        layer + 1, loc_row_offset, 0) /*+ (layer + 1) * M * WIDTH + layer_offset_A*/);
                    }

                    MAD<TM, TK>(As, weights.GetMatrixPointer(n_hidden_layers), Cs);

                    // activate
                    applyActivation<output_activation, TM, TK>(Cs, As);

                    // save to HBM
                    if constexpr (!INFERENCE)
                        storeRow<TM, TK, cache_hint::uncached, cache_hint::write_back>(
                            As, intermediate_output.GetElementPointer(
                                    n_hidden_layers + 1, loc_row_offset,
                                    0) /*+ (n_hidden_layers + 1) * M * WIDTH + layer_offset_A*/);
                    else if constexpr (INFERENCE) // storing at the beginning since no intermediate results
                        storeRow<TM, TK, cache_hint::uncached, cache_hint::write_back>(
                            As, intermediate_output.GetElementPointer(0, loc_row_offset, 0));
                });
        });

        return {e};
    }
};

extern template class EsimdKernels<bf16, 16, 16, 16, Activation::None, Activation::None>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<bf16, 16, 16, 16, Activation::Sigmoid, Activation::Sigmoid>;

extern template class EsimdKernels<bf16, 32, 32, 32, Activation::None, Activation::None>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<bf16, 32, 32, 32, Activation::Sigmoid, Activation::Sigmoid>;

extern template class EsimdKernels<bf16, 64, 64, 64, Activation::None, Activation::None>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<bf16, 64, 64, 64, Activation::Sigmoid, Activation::Sigmoid>;

extern template class EsimdKernels<bf16, 128, 128, 128, Activation::None, Activation::None>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<bf16, 128, 128, 128, Activation::Sigmoid, Activation::Sigmoid>;

extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::None, Activation::None>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 16, 16, 16, Activation::Sigmoid, Activation::Sigmoid>;

extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::None, Activation::None>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 32, 32, 32, Activation::Sigmoid, Activation::Sigmoid>;

extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::None, Activation::None>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 64, 64, 64, Activation::Sigmoid, Activation::Sigmoid>;

extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::None, Activation::None>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::None, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::None, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::ReLU, Activation::None>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::ReLU, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::ReLU, Activation::Sigmoid>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::Sigmoid, Activation::None>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::Sigmoid, Activation::ReLU>;
extern template class EsimdKernels<sycl::half, 128, 128, 128, Activation::Sigmoid, Activation::Sigmoid>;

} // namespace esimd
} // namespace kernels
} // namespace tinydpcppnn
