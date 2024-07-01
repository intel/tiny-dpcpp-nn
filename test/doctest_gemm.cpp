/**
 * @file doctest_gemm.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief file to test gemm functions and functionalities
 * @version 0.1
 * @date 2024-05-28
 * 
 * Copyright (c) 2024 Intel Corporation
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "doctest/doctest.h"

#include "gemm.h"
#include "result_check.h"

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T, int WIDTH>
void TestGemm(const size_t M, const int nbatches) {
    sycl::queue q;

    T * A = sycl::malloc_device<T>(nbatches * M * WIDTH, q);
    T * B = sycl::malloc_device<T>(nbatches * M*WIDTH, q);
    T * C = sycl::malloc_device<T>(nbatches * WIDTH*WIDTH, q);

    q.parallel_for(nbatches * M * WIDTH, [=](auto idx) { A[idx] = static_cast<T>(0.5); }).wait();
    q.parallel_for(nbatches * M * WIDTH, [=](auto idx) { B[idx] = static_cast<T>(1.0); }).wait();
    
    Gemm<T, WIDTH> gemm;
    gemm.batched(M, nbatches, A, B, C, q);

    q.wait();

    std::vector<T> C_host(nbatches * WIDTH*WIDTH);
    q.memcpy(C_host.data(), C, nbatches * WIDTH*WIDTH * sizeof(T)).wait();

    CHECK(isVectorWithinTolerance(C_host, 0.5*M, 1.0e-4));
    std::cout << C_host[0] << ", " << C_host[1] << ", " << C_host[2] << ", " << std::endl;

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    
}

TEST_CASE("Gemm 1024,1,16,bf16") {

    const size_t M = 16;
    const int nbatches = 1;
    constexpr int WIDTH = 16;
    
    TestGemm<bf16, WIDTH>(M, nbatches);
}

TEST_CASE("Gemm 1024,1,16,bf16") {

    const size_t M = 1024;
    const int nbatches = 1;
    constexpr int WIDTH = 16;
    
    TestGemm<bf16, WIDTH>(M, nbatches);
}

TEST_CASE("Gemm 1024,4,16,bf16") {

    const size_t M = 1024;
    const int nbatches = 4;
    constexpr int WIDTH = 16;
    
    TestGemm<bf16, WIDTH>(M, nbatches);
}

TEST_CASE("Gemm 1024,4,64,bf16") {

    const size_t M = 1024;
    const int nbatches = 4;
    constexpr int WIDTH = 16;
    
    TestGemm<bf16, WIDTH>(M, nbatches);
}
