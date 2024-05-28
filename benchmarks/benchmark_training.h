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

#include "SwiftNetMLP.h"
#include "common.h"
#include "common_benchmarks.h"
#include "mpi.h"
#include "result_check.h"
#include "trainer.h"

/// benchmarking function with input width = width = output width
/// Note that this is not meant to test the correctness, only perf.
/// Correctness is checked with the tests in the 'test' directory
template <typename T, int WIDTH>
double benchmark_training(const size_t batch_size, const int n_hidden_layers, const int n_iterations, sycl::queue &q) {

    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Training (forw+backw, no opt, no loss)", batch_size, WIDTH,
                                                          n_hidden_layers, sizeof(T), type_to_string<T>(), q);

    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;

    DeviceMatrix<T> inputs(batch_size, input_width, q);
    DeviceMatrices<T> outputs_backw(n_hidden_layers + 1, input_width, WIDTH, WIDTH, WIDTH, WIDTH, output_width, q);
    DeviceMatrix<T> losses(batch_size, output_width, q);
    // includes input and output (thus +1 of the back interm)
    DeviceMatrices<T> out_inter_forw(n_hidden_layers + 2, batch_size, input_width, batch_size, WIDTH, batch_size,
                                     output_width, q);
    DeviceMatrices<T> out_inter_backw(n_hidden_layers + 1, batch_size, WIDTH, batch_size, WIDTH, batch_size,
                                      output_width, q);

    const T input_val = static_cast<T>(sqrt(1.0/batch_size));
    inputs.fill(input_val);
    outputs_backw.fill(0);
    const T losses_val = static_cast<T>(sqrt(1.0/batch_size));
    losses.fill(losses_val);

    // need a factory here for different widths
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::None, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    std::vector<T> new_weights(network.get_weights_matrices().nelements(), 1.0 / WIDTH);
    network.set_weights_matrices(new_weights);
    q.wait();

    Trainer<T> train(&network);

    constexpr int n_iterations_warmup = 5;
    // Do a warmup loop, not benched.
    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        train.training_step(inputs, outputs_backw, losses, out_inter_forw, out_inter_backw, {});
        q.wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin_time = std::chrono::steady_clock::now();
    std::vector<sycl::event> dependencies;
    for (int iter = 0; iter < n_iterations; iter++) {
        dependencies =
            train.training_step(inputs, outputs_backw, losses, out_inter_forw, out_inter_backw, dependencies);
    }
    q.wait();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end_time = std::chrono::steady_clock::now();

    double gflops = tinydpcppnn::benchmarks::common::WritePerformanceDataTraining(
        begin_time, end_time, batch_size, WIDTH, n_hidden_layers, n_iterations, sizeof(T));

    MPI_Barrier(MPI_COMM_WORLD);

    // Now do a simple correctness check
    // TODO: check all the elements in the interm forw array.
    // expect that the output of the backward pass is 0 since losses are set to 0
    isVectorWithinTolerance(out_inter_forw.copy_to_host(), input_val, 1.0e-4);
    std::cout << "out_inter_forw done." << std::endl;
    isVectorWithinTolerance(out_inter_backw.copy_to_host(), losses_val, 1.0e-4);
    std::cout << "out_inter_backw done." << std::endl;

    auto outputs_backw_host = outputs_backw.copy_to_host();
    isVectorWithinTolerance(outputs_backw_host, 1.0, 1.0e-4);
    std::cout << outputs_backw_host[0] << ", " << outputs_backw_host[1] << ", " << outputs_backw_host[2] << ", "
              << outputs_backw_host[3] << std::endl;
    std::cout << std::endl;

    return gflops;
}
