/**
 * @file benchmark_inference.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a templated inference benchmark function.
 * TODO: make this a class and derived it from a base class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "SwiftNetMLP.h"
#include "common.h"
#include "common_benchmarks.h"
#include "mpi.h"
#include "result_check.h"

/// benchmarking function with input width = width = output width
template <typename T, int WIDTH>
double benchmark_inference(const size_t batch_size, const int n_hidden_layers, const int n_iterations, sycl::queue &q) {

    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Inference", batch_size, WIDTH, n_hidden_layers, sizeof(T),
                                                          type_to_string<T>(), q);

    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;

    DeviceMatrix<T> inputs(batch_size, input_width, q);
    DeviceMatrix<T> output(batch_size, output_width, q);

    const T input_val = static_cast<T>(0.1);
    inputs.fill(input_val);
    output.fill(0);

    // need a factory here for different widths
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    std::vector<T> new_weights(network.get_weights_matrices().nelements(), 1.0 / WIDTH);
    network.set_weights_matrices(new_weights);

    constexpr int n_iterations_warmup = 5;
    // Do a warmup loop, not benched.
    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        network.inference(inputs, output, {});
        q.wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin_time = std::chrono::steady_clock::now();
    std::vector<sycl::event> dependencies;
    for (int iter = 0; iter < n_iterations; iter++) {
        dependencies = network.inference(inputs, output, dependencies);
    }
    q.wait();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end_time = std::chrono::steady_clock::now();

    double gflops = tinydpcppnn::benchmarks::common::WritePerformanceDataInference(
        begin_time, end_time, batch_size, WIDTH, n_hidden_layers, n_iterations, sizeof(T));

    MPI_Barrier(MPI_COMM_WORLD);
    isVectorWithinTolerance(output.copy_to_host(), input_val, 1.0e-4);
    std::cout << std::endl;

    return gflops;
}
