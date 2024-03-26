/**
 * @file common_benchmarks.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Common functions used in the benchmarks. Mostly performance computations and output to console.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "mpi.h"
#include <chrono>
#include <cmath>
#include <iostream>

#include "SwiftNetMLP.h"

extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 16>;
extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 32>;
extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 64>;
extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 128>;

extern template class SwiftNetMLP<sycl::half, 16>;
extern template class SwiftNetMLP<sycl::half, 32>;
extern template class SwiftNetMLP<sycl::half, 64>;
extern template class SwiftNetMLP<sycl::half, 128>;

/// TODO: take it apart into a .h and a .cpp

namespace tinydpcppnn {
namespace benchmarks {
namespace common {

struct PerformanceData {
    /// TODO: WIDTH, layers, data type, etc. are missing in this. Add those in the struct and print out in
    /// benchmark_all.cpp and  benchmark_all_torch.cpp

    int batch_size;
    double training_gflops;
    double inference_gflops;
};

/// Put all the common functionalities for the performance benchmarks here.
void WriteBenchmarkHeader(const std::string &str, const size_t batch_size, const int WIDTH, const int n_hidden_layers,
                          const int typesize, const std::string &name_of_type, sycl::queue &q) {

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        std::cout << str << std::endl;
        std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::cout << "n_hidden_layers = " << n_hidden_layers << ", WIDTH = " << WIDTH << ", batch_size = " << batch_size
                  << ", typename = " << name_of_type << ", type size = " << typesize << " bytes" << std::endl
                  << "MPI world_size = " << world_size << std::endl;
    }
}

void WritePerformanceData(const int n_iterations, const double time, const double oi, const double bw,
                          const double tp) {
    std::cout << "Finished benchmark." << std::endl;
    std::cout << "#Iterations =\t\t" << n_iterations << std::endl;
    std::cout << "Time =\t\t\t" << time << " \ts" << std::endl;
    std::cout << "AI (infinite $) =\t" << oi << " \tflops/byte" << std::endl;
    std::cout << "BW (infinite $) =\t" << bw << " \tGB/s" << std::endl;
    std::cout << "Throughput =\t\t" << tp << " \tGflops/s" << std::endl;
}

// TODO: consolidate WritePerformanceData inference and training and just write functions
// which return the corresponding flops and byte
double WritePerformanceDataInference(std::chrono::time_point<std::chrono::steady_clock> begin,
                                     std::chrono::time_point<std::chrono::steady_clock> end, const size_t batch_size,
                                     const int WIDTH, const int n_hidden_layers, const int n_iterations,
                                     const int typesize) {

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    const double flops =
        world_size * 2.0 * (double)batch_size * (double)WIDTH * (double)WIDTH * (n_hidden_layers + 1) * n_iterations;
    // flops / ns = gflops/s
    const double gflops_per_s = flops / elapsed_time;

    if (world_rank == 0) {
        // load A once, store A once, load B for every layer per iteration.
        const double bytes_loaded_and_stored =
            typesize * world_size * (2.0 * batch_size * WIDTH + WIDTH * WIDTH * (n_hidden_layers + 1)) * n_iterations;
        const double hbm_bandwidth_GB_per_s = bytes_loaded_and_stored / elapsed_time;
        // arithmetic intensity
        const double oi = flops / bytes_loaded_and_stored;

        WritePerformanceData(n_iterations, elapsed_time * 1.0e-9, oi, hbm_bandwidth_GB_per_s, gflops_per_s);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return gflops_per_s;
}

double WritePerformanceDataTraining(std::chrono::time_point<std::chrono::steady_clock> begin,
                                    std::chrono::time_point<std::chrono::steady_clock> end, const size_t batch_size,
                                    const int WIDTH, const int n_hidden_layers, const int n_iterations,
                                    const int typesize) {

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    // we have three matrix multiplications of size M*K*N per layer per iteration in the training
    const double flops = 3 * world_size * 2.0 * (double)batch_size * (double)WIDTH * (double)WIDTH *
                         (n_hidden_layers + 1) * n_iterations;
    const double gflops_per_s = flops / elapsed_time;

    if (world_rank == 0) {

        const double bytes_loaded_and_stored =
            typesize * world_size * n_iterations *
            (batch_size * WIDTH + 2 * (WIDTH * WIDTH + batch_size * WIDTH) * (n_hidden_layers + 1) +
             (2 * batch_size * WIDTH + WIDTH * WIDTH) * (n_hidden_layers + 1));
        const double hbm_bandwidth_GB_per_s = bytes_loaded_and_stored / elapsed_time;
        // arithmetic intensity
        const double oi = flops / bytes_loaded_and_stored;

        WritePerformanceData(n_iterations, elapsed_time * 1.0e-9, oi, hbm_bandwidth_GB_per_s, gflops_per_s);
        // Return the gflops_per_s value instead of void.
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return gflops_per_s;
}

} // namespace common
} // namespace benchmarks
} // namespace tinydpcppnn