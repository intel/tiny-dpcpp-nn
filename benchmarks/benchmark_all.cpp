// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <sycl/sycl.hpp>

#include "benchmark_inference.h"
#include "benchmark_training.h"
#include "mpi.h"

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T, int WIDTH>
void benchmark_training_and_inference(const size_t batch_size, const int n_hidden_layers, const int n_iterations,
                                      sycl::queue &q, double &gflops_training, double &gflops_inference) {
    gflops_training = benchmark_training<T, WIDTH>(batch_size, n_hidden_layers, n_iterations, q);
    q.wait();
    gflops_inference = benchmark_inference<T, WIDTH>(batch_size, n_hidden_layers, n_iterations, q);
    q.wait();
}

template <typename T, int WIDTH> void benchmark_all(sycl::queue &q, int test_over_batch_size = 0) {
    int n_hidden_layers = 4;
    int iterations = 100;
    int batch_size;
    std::vector<tinydpcppnn::benchmarks::common::PerformanceData> perf_data;

    int world_rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double gflops_training, gflops_inference;
    int batch_size_offset =
        1 - size; // if MPI size is 2 (2 tiles on PVC), then we need to run 2^(batch_size + batch_size_offset) per tile
    std::cout << "MPI size: " << size << ", thus batch size on each MPI rank: " << batch_size_offset << std::endl;
    if (test_over_batch_size) {
        // benchmark training over all batch_size
        if (world_rank == 0) {
            std::cout << "=========================Benchmark throughput over batch sizes========================="
                      << std::endl;
        }
        for (int power = 10; power < 22; power++) {
            batch_size = 1 << (power + batch_size_offset);
            benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q, gflops_training,
                                                       gflops_inference);

            // Collect the performance data instead of printing it directly
            perf_data.push_back({power, gflops_training, gflops_inference});
        }
    }

    else {

        iterations = 1000; // all benchmarks run 1000 iters
        n_hidden_layers = 11;
        batch_size =
            1 << (17 + batch_size_offset); // batch size one less, because MPI does 2 tiles, thus half batch size.
        if (world_rank == 0) {
            std::cout
                << "=================================Benchmark of n_hidden_layers 11================================="
                << std::endl;
        }
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q, gflops_training,
                                                   gflops_inference);

        // Collect the performance data instead of printing it directly
        perf_data.push_back({batch_size, gflops_training, gflops_inference});
        // Image compression
        n_hidden_layers = 2;
        // batch_size = {2304 * 3072}; // resolution of image
        // batch size was 23 before, but arc and dGPU don't have enough memory, thus 22.
        batch_size =
            1 << (22 + batch_size_offset); // batch size one less, because MPI does 2 tiles, thus half batch size.
        if (world_rank == 0) {
            std::cout << "=================================Image compression================================="
                      << std::endl;
        }
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q, gflops_training,
                                                   gflops_inference);

        // Collect the performance data instead of printing it directly
        perf_data.push_back({batch_size, gflops_training, gflops_inference});
        // PINNs
        n_hidden_layers = 5;
        if (world_rank == 0) {
            std::cout << "=================================PINNs=================================" << std::endl;
        }
        batch_size =
            1 << (17 + batch_size_offset); // batch size one less, because MPI does 2 tiles, thus half batch size.
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q, gflops_training,
                                                   gflops_inference);

        // Collect the performance data instead of printing it directly
        perf_data.push_back({batch_size, gflops_training, gflops_inference});
        // NeRF
        n_hidden_layers = 4;
        if (world_rank == 0) {
            std::cout << "=================================NeRF=================================" << std::endl;
        }
        batch_size =
            1 << (20 + batch_size_offset); // batch size one less, because MPI does 2 tiles, thus half batch size.
        benchmark_training_and_inference<T, WIDTH>(batch_size, n_hidden_layers, iterations, q, gflops_training,
                                                   gflops_inference);
        // Collect the performance data instead of printing it directly
        perf_data.push_back({batch_size, gflops_training, gflops_inference});
    }

    // At the end, print the collected performance data in a table format
    if (world_rank == 0) {
        std::cout << "| Batch Size | Training FLOPS/s | Inference FLOPS/s |\n";
        std::cout << "|------------|------------------|-------------------|\n";
        for (const auto &data : perf_data) {
            std::cout << " " << data.batch_size << " , " << data.training_gflops * 1e9 << " , "
                      << data.inference_gflops * 1e9 << "\n";
        }

        // Save the data
        if (test_over_batch_size) {
            // Create three separate vectors to store the data
            std::vector<int> batch_sizes;
            std::vector<double> training_throughputs;
            std::vector<double> inference_throughputs;

            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            // we actually are calculating flops/elements, and batchsize * iteration cancels out
            double flops = world_size * 2.0 * (double)WIDTH * (double)WIDTH * (n_hidden_layers + 1) /
                           1e9; // dividing by 1e9, because the training_flops are in gflops

            // Extract the performance data into the vectors
            for (const auto &data : perf_data) {
                batch_sizes.push_back(data.batch_size);
                training_throughputs.push_back(data.training_gflops / flops);
                inference_throughputs.push_back(data.inference_gflops / flops);
            }

            // Create one JSON object
            nlohmann::json json_obj;

            json_obj["batch_sizes"] = batch_sizes;
            json_obj["training_throughputs"] = training_throughputs;
            json_obj["inference_throughputs"] = inference_throughputs;

            // Construct the JSON filename (make sure WIDTH has been defined appropriately)
            std::string filename = "../benchmarks/results/bench_result_dpcpp_width" + std::to_string(WIDTH) + ".json";
            // Save the JSON object to a file
            std::ofstream ostream(filename);
            if (!ostream.is_open()) {
                std::cout << "\n Failed to open output file " << filename;
            } else {
                ostream << json_obj.dump(4); // Serialize the JSON object with an indentation of 4 spaces
                ostream.close();
            }
        }
    }
}
int main() {
    try {
        MPI_Init(NULL, NULL);
        sycl::queue q(sycl::gpu_selector_v);
        // ----------Benchmark for different workloads----------

        std::cout << "Sycl::half, width 16" << std::endl;
        benchmark_all<sycl::half, 16>(q, 0);

        std::cout << "Sycl::half, width 32" << std::endl;
        benchmark_all<sycl::half, 32>(q, 0);

        std::cout << "Sycl::half, width 64" << std::endl;
        benchmark_all<sycl::half, 64>(q, 0);

        std::cout << "Sycl::half, width 128" << std::endl;
        benchmark_all<sycl::half, 128>(q, 0);

        // ----------Benchmark over batch sizes----------

        std::cout << "Sycl::half, width 16" << std::endl;
        benchmark_all<sycl::half, 16>(q, 1);

        std::cout << "Sycl::half, width 32" << std::endl;
        benchmark_all<sycl::half, 32>(q, 1);

        std::cout << "Sycl::half, width 64" << std::endl;
        benchmark_all<sycl::half, 64>(q, 1);

        std::cout << "Sycl::half, width 128" << std::endl;
        benchmark_all<sycl::half, 128>(q, 1);

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