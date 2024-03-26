/**
 * @file benchmark_training.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a main file for trainign benchmarks. Runs multiple training test cases.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <sycl/sycl.hpp>

#include "benchmark_training.h"
#include "mpi.h"

using bf16 = sycl::ext::oneapi::bfloat16;

int main() {
    try {
        MPI_Init(NULL, NULL);
        sycl::queue q(sycl::gpu_selector_v);

        benchmark_training<bf16, 64>(1 << 22, 4, 100, q);
        q.wait();
        benchmark_training<sycl::half, 64>(1 << 22, 4, 100, q);
        q.wait();
        benchmark_training<bf16, 32>(1 << 22, 4, 100, q);
        q.wait();
        benchmark_training<bf16, 16>(1 << 22, 4, 100, q);
        q.wait();
        for (int iter = 10; iter < 24; iter++) {
            benchmark_training<bf16, 64>(1 << iter, 4, 100, q);
            q.wait();
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
