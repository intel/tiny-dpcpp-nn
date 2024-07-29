/**
 * @file doctest_kernel_esimd.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Class which tests the esimd kernels.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"

#include "kernel_esimd.h"

#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;
using sycl::ext::intel::experimental::esimd::cache_hint;
using namespace tinydpcppnn::kernels::esimd;
using namespace sycl::ext::intel::esimd;

template <int M, int N, int TK, typename T> void TestLoadRow(sycl::queue &q) {
    constexpr int nElems = M * N;
    T *in = sycl::malloc_device<T>(nElems, q);
    T *out = sycl::malloc_device<T>(nElems, q);
    std::vector<T> in_host(nElems);
    std::vector<T> out_host(nElems);
    for (int iter = 0; iter < nElems; iter++) {
        in_host[iter] = static_cast<T>(iter);
    }

    q.memcpy(in, in_host.data(), sizeof(T) * nElems).wait();

    q.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
         simd<T, nElems> tmp;
         EsimdKernels<T, N, N, N, Activation::ReLU, Activation::None>::template loadRow<M, TK, cache_hint::none,
                                                                                        cache_hint::none>(in, tmp);
         tmp.copy_to(out);
     }).wait();

    q.memcpy(out_host.data(), out, sizeof(T) * nElems).wait();

    for (int iter = 0; iter < nElems; iter++) {
        const int block = iter / (M * TK);
        const int row = (iter % (M * TK)) / TK;
        const int elem = iter % TK;
        CHECK(out_host[iter] == in_host[elem + block * TK + row * N]);
    }

    sycl::free(in, q);
    sycl::free(out, q);
}

template <int M, int N, int TK, typename T> void TestLoadStoreRow(sycl::queue &q) {
    constexpr int nElems = M * N;
    T *in = sycl::malloc_device<T>(nElems, q);
    T *out = sycl::malloc_device<T>(nElems, q);
    std::vector<T> in_host(nElems);
    std::vector<T> out_host(nElems);
    for (int iter = 0; iter < nElems; iter++) {
        in_host[iter] = static_cast<T>(iter);
    }

    q.memcpy(in, in_host.data(), sizeof(T) * nElems).wait();

    q.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
         simd<T, nElems> tmp;
         EsimdKernels<T, N, N, N, Activation::ReLU, Activation::None>::template loadRow<M, TK, cache_hint::none,
                                                                                        cache_hint::none>(in, tmp);
         EsimdKernels<T, N, N, N, Activation::ReLU, Activation::None>::template storeRow<M, TK, cache_hint::none,
                                                                                         cache_hint::none>(tmp, out);
     }).wait();

    q.memcpy(out_host.data(), out, sizeof(T) * nElems).wait();

    for (int iter = 0; iter < nElems; iter++) {
        const int block = iter / (M * TK);
        const int row = (iter % (M * TK)) / TK;
        const int elem = iter % TK;
        CHECK(out_host[iter] == in_host[iter]);
    }

    sycl::free(in, q);
    sycl::free(out, q);
}

template <int TM, int TK, int TN, int N, typename T> void TestReBlock(sycl::queue &q) {
    constexpr int nElems = TM * N;
    T *out = sycl::malloc_device<T>(nElems, q);
    std::vector<T> out_host(nElems);

    q.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
         simd<T, nElems> tmp(0, 1); // block-major with TMxTN blocks
         simd<T, nElems> dst(0);
         EsimdKernels<T, N, N, N, Activation::ReLU, Activation::None>::template reBlock<TM, TK, TN>(tmp, dst);
         dst.copy_to(out); // block-major with TMxTK blocks
     }).wait();

    q.memcpy(out_host.data(), out, sizeof(T) * nElems).wait();
    for (int iter = 0; iter < TM; iter++) {
        for (int coliter = 0; coliter < N; coliter++) {
            std::cout << out_host[iter * N + coliter] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

    if constexpr (TN == TK) {
        for (int iter = 0; iter < nElems; iter++) {
            bool tmp = (int)out_host[iter] == static_cast<int>(iter);
            CHECK(tmp);
            if (!tmp) return;
        }
    } else if constexpr (TK == TN / 2) { // TK == 8 and TN == 16
        constexpr int ratio = TN / TK;   // in how many matrices are we splitting
        for (int iter = 0; iter < nElems; iter++) {
            const int block_old = iter / (TM * TN);
            const int new_sub_block = (iter % (TM * TN)) / (TM * TK);
            const int new_row = (iter % (TM * TK)) / TK;
            const int new_col = (iter % (TM * TK)) % TK;
            const int val = block_old * (TM * TN) + new_sub_block * TK + new_row * TN + new_col;

            // std::cout << (int)out[iter] << ", " << iter << ", " << val << std::endl;
            bool tmp = ((int)out_host[iter] == static_cast<int>(val));
            CHECK(tmp);
            if (!tmp) return;
        }
    } else if constexpr (TN < TK) {
        constexpr int ratio = TK / TN; // how many matrices are we merging.
        for (int iter = 0; iter < nElems; iter++) {
            const int block_new = iter / (TM * TK);
            const int old_sub_block = ((iter % (TM * TK)) / TN) % ratio;
            const int old_row = (iter % (TM * TK)) / TK;
            const int old_col = (iter % (TM * TK)) % TN;
            const int val = block_new * TM * TK + old_sub_block * TM * TN + old_row * TN + old_col;

            // std::cout << (int)out[iter] << ", " << iter << ", " << val << "; " << block_new << ", " << old_sub_block
            //           << ", " << old_row << ", " << old_col << std::endl;
            bool tmp = ((int)out_host[iter] == static_cast<int>(val));
            CHECK(tmp);
            if (!tmp) return;
        }

    } else
        throw std::logic_error("This combination cannot exist.");

    sycl::free(out, q);
}

TEST_CASE("LoadRow") {

    sycl::queue q(sycl::gpu_selector_v);

    SUBCASE("load row bf16 N64 M8 TK16") { TestLoadRow<8, 64, 16, bf16>(q); }
    SUBCASE("load row half N64 M8 TK16") { TestLoadRow<8, 64, 16, sycl::half>(q); }
    SUBCASE("load row bf16 N16 M8 TK16") { TestLoadRow<8, 16, 16, bf16>(q); }
    SUBCASE("load row bf16 N32 M8 TK16") { TestLoadRow<8, 32, 16, bf16>(q); }
    SUBCASE("load row bf16 N128 M8 TK16") { TestLoadRow<8, 128, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M1 TK16") { TestLoadRow<1, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M2 TK16") { TestLoadRow<2, 64, 16, bf16>(q); }
    SUBCASE("load row bf16 N64 M4 TK16") { TestLoadRow<4, 64, 16, bf16>(q); }
}

TEST_CASE("LoadStoreRow") {

    sycl::queue q(sycl::gpu_selector_v);

    SUBCASE("loadstore row bf16 N64 M8 TK16") { TestLoadStoreRow<8, 64, 16, bf16>(q); }
    SUBCASE("loadstore row half N64 M8 TK16") { TestLoadStoreRow<8, 64, 16, sycl::half>(q); }
    SUBCASE("loadstore row bf16 N16 M8 TK16") { TestLoadStoreRow<8, 16, 16, bf16>(q); }
    SUBCASE("loadstore row bf16 N32 M8 TK16") { TestLoadStoreRow<8, 32, 16, bf16>(q); }
    SUBCASE("loadstore row bf16 N128 M8 TK16") { TestLoadStoreRow<8, 128, 16, bf16>(q); }
    SUBCASE("loadstore row bf16 N64 M1 TK16") { TestLoadStoreRow<1, 64, 16, bf16>(q); }
    SUBCASE("loadstore row bf16 N64 M2 TK16") { TestLoadStoreRow<2, 64, 16, bf16>(q); }
    SUBCASE("loadstore row bf16 N64 M4 TK16") { TestLoadStoreRow<4, 64, 16, bf16>(q); }
}

TEST_CASE("reBlock") {
    sycl::queue q(sycl::gpu_selector_v);

    // SUBCASE("Equal dims 8") { TestReBlock<8, XMXTn::TN, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("Equal dims 4") { TestReBlock<4, XMXTn::TN, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("Equal dims 2") { TestReBlock<2, XMXTn::TN, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("Equal dims 1") { TestReBlock<1, XMXTn::TN, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("2x dims 8") { TestReBlock<8, XMXTn::TN * 2, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("2x dims 4") { TestReBlock<4, XMXTn::TN * 2, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("2x dims 2") { TestReBlock<2, XMXTn::TN * 2, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("2x dims 1") { TestReBlock<1, XMXTn::TN * 2, XMXTn::TN, 64, int16_t>(q); }
    // SUBCASE("4x dims") { TestReBlock<8, XMXTn::TN * 4, XMXTn::TN, 64, int16_t>(q); }
    // if constexpr (XMXTn::TN == 16) {
    //     SUBCASE(".5x dims") { TestReBlock<8, XMXTn::TN / 2, XMXTn::TN, 64, int16_t>(q); }
    // }
    SUBCASE("8x8") { TestReBlock<8, 8, 8, 64, int16_t>(q); }
    SUBCASE("16x8") { TestReBlock<8, 16, 8, 64, int16_t>(q); }
    SUBCASE("32x8") { TestReBlock<8, 32, 8, 64, int16_t>(q); }
    SUBCASE("64x8") { TestReBlock<8, 64, 8, 64, int16_t>(q); }
    SUBCASE("8x16") { TestReBlock<8, 8, 16, 64, int16_t>(q); }
    SUBCASE("16x16") { TestReBlock<8, 16, 16, 64, int16_t>(q); }
    SUBCASE("32x16") { TestReBlock<8, 32, 16, 64, int16_t>(q); }
    SUBCASE("64x16") { TestReBlock<8, 64, 16, 64, int16_t>(q); }
}
