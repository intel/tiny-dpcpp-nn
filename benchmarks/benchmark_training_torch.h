/**
 * @file benchmark_training.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a templated benchmark training function for tests.
 * TODO: implement this as a class which is derived from a benchmark base class.
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

#include "common_benchmarks.h"
#include "mpi.h"
#include "trainer_torch.h"

/// benchmarking function with input width = width = output width
/// Note that this is not meant to test the correctness, only perf.
/// Correctness is checked with the tests in the 'test' directory
template <typename T, int WIDTH>
double benchmark_training(const size_t batch_size, const int n_hidden_layers, const int n_iterations) {

    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr float weight_val = 1.0 / WIDTH;

    tnn::NetworkModule<T, WIDTH> network(input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                         false);

    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Training (forw+backw, no opt, no loss)", batch_size, WIDTH,
                                                          n_hidden_layers, sizeof(T), type_to_string<T>(),
                                                          network.get_queue());

    torch::Tensor input = torch::ones({(int)batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16);
    torch::Tensor dL_doutput =
        torch::ones({(int)batch_size, output_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16);

    Trainer<T> train(&network, weight_val);

    constexpr int n_iterations_warmup = 5;
    // Do a warmup loop, not benched.
    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        train.training_step(input, dL_doutput);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin_time = std::chrono::steady_clock::now();
    std::vector<sycl::event> dependencies;
    torch::Tensor output;
    for (int iter = 0; iter < n_iterations; iter++) {
        output = train.training_step(input, dL_doutput);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end_time = std::chrono::steady_clock::now();

    double gflops = tinydpcppnn::benchmarks::common::WritePerformanceDataTraining(
        begin_time, end_time, batch_size, WIDTH, n_hidden_layers, n_iterations, sizeof(T));

    MPI_Barrier(MPI_COMM_WORLD);

    const float output_ref = std::pow(weight_val * WIDTH, n_hidden_layers + 1);
    bool all_values_correct = torch::allclose(output, torch::full_like(output, output_ref), 1e-5);
    if (all_values_correct) {
        std::cout << "All values in the tensor are correct." << std::endl;
    } else {
        std::cout << "Not all values in the tensor are correct." << std::endl;
    }

    return gflops;
}
