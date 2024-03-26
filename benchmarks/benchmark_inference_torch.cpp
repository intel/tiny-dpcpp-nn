/**
 * @file benchmark_inference.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief inference benchmarks. Implements a main which runs several inference cases.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <iostream>
#include <sycl/sycl.hpp>

#include "benchmark_inference_torch.h"
#include "mpi.h"

using bf16 = sycl::ext::oneapi::bfloat16;

/**
 * @brief Main which calls multiple inference tests.
 *
 * @return int 0 if everything is alright. If an exception is caught 1 or 2
 */
int main() {
    try {
        MPI_Init(NULL, NULL);
        // sycl::queue q(sycl::gpu_selector_v);

        benchmark_inference<bf16, 64>(1 << 22, 4, 1000);

        benchmark_inference<sycl::half, 64>(1 << 22, 4, 1000);

        benchmark_inference<bf16, 16>(1 << 22, 4, 1000);

        benchmark_inference<bf16, 32>(1 << 22, 4, 1000);

        benchmark_inference<bf16, 128>(1 << 22, 4, 1000);

        benchmark_inference<sycl::half, 128>(1 << 22, 4, 1000);

        for (int iter = 16; iter < 25; iter++) {
            benchmark_inference<bf16, 64>(1 << iter, 4, 100);
        }

        for (int iter = 10; iter < 25; iter++) {
            benchmark_inference<bf16, 16>(1 << iter, 4, 1000);
        }

        for (int iter = 2; iter < 20; iter++) {
            benchmark_inference<bf16, 64>(1 << 22, iter, 1000);
        }
        MPI_Finalize();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cout << "Caught some undefined exception." << std::endl;
        return 2;
    }

    return 0;
}
