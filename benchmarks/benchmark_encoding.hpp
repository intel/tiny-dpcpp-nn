#pragma once

#include "encoding_factory.h"
#include "common.h"
#include "common_benchmarks.h"
#include "mpi.h"
#include "result_check.h"
#include "DeviceMatrix.h"

using json = nlohmann::json;

/// benchmarking function with input width = width = output width
template <typename T, int WIDTH>
double benchmark_encoding(const size_t batch_size, const size_t input_width, const int n_iterations, 
    const json &config, sycl::queue &q) {

    constexpr int output_width = WIDTH;

    DeviceMatrix<T> inputs(batch_size, input_width, q);
    DeviceMatrix<T> output(batch_size, output_width, q);

    const T input_val = static_cast<T>(0.1);
    inputs.fill(input_val);
    output.fill(0);

    // need a factory here for different widths
    std::cout << "Creating encoding" << std::endl;
    auto encoding = create_encoding<T>(config, q, output_width);

    auto output_view = output.GetView();
    std::cout << "Warmup" << std::endl;
    encoding->forward_impl(inputs.GetView(), &output_view);
    q.wait();

    std::cout << "Starting Iter" << std::endl;
    const auto begin_time = std::chrono::steady_clock::now();
    for (int iter = 0; iter < n_iterations; iter++) {
        encoding->forward_impl(inputs.GetView(), &output_view);
        q.wait();
    }
    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "Elapsed time per iter = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count()/(double)n_iterations << " ms\n";

    return 0;
}