/**
 * @file compare_exp.cpp
 * @brief Compare esimd::exp and std::exp for accuracy
 * @version 0.1
 * @date 2024-07-16
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"
#include <cmath>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>

void compare_exp(sycl::queue &q, float start, float end, float step) {
    int nElems = static_cast<int>((end - start) / step) + 1;
    std::vector<float> x_host(nElems), esimd_exp_host(nElems), std_exp_host(nElems);

    // Initialize input values
    for (int i = 0; i < nElems; ++i) {
        x_host[i] = start + i * step;
        std_exp_host[i] = std::exp(-x_host[i]);
    }

    // Allocate device memory
    float *x = sycl::malloc_device<float>(nElems, q);
    float *esimd_exp = sycl::malloc_device<float>(nElems, q);

    // Copy input data to device
    q.memcpy(x, x_host.data(), sizeof(float) * nElems).wait();

    // Run ESIMD kernel

    q.parallel_for(sycl::range<1>(nElems), [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
         sycl::ext::intel::esimd::simd<float, 1> val;
         val.copy_from(x + i);
         sycl::ext::intel::esimd::simd<float, 1> result = sycl::ext::intel::esimd::exp(-val);
         result.copy_to(esimd_exp + i);
     }).wait();

    // Copy results back to host
    q.memcpy(esimd_exp_host.data(), esimd_exp, sizeof(float) * nElems).wait();

    // Compare the results
    float max_error = 0.0f;
    float total_error = 0.0f;

    for (int i = 0; i < nElems; ++i) {
        float error = std::abs(esimd_exp_host[i] - std_exp_host[i]);
        std::cout << "for x: " << x_host[i] << ", error: " << error << ", with esimd: " << esimd_exp_host[i]
                  << ", std: " << std_exp_host[i] << std::endl;
        total_error += error;
        if (error > max_error) {
            max_error = error;
        }
    }

    float average_error = total_error / nElems;

    std::cout << "Maximum error: " << max_error << std::endl;
    std::cout << "Average error: " << average_error << std::endl;

    // Free device memory
    sycl::free(x, q);
    sycl::free(esimd_exp, q);
}

TEST_CASE("Compare exp functions") {
    sycl::queue q(sycl::gpu_selector_v);

    SUBCASE("Compare exp from -20 to 20 with step 1e-3") { compare_exp(q, -10.0f, 10.0f, 0.1f); }
}
