/**
 * @file doctest_torch_network.cpp
 * @author Kai Yuan (kai.yuan@intel.com)
 * @brief Tests for the Torch and tnn_api.h.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"
#include "l2.h"
#include "mlp.h"
#include "result_check.h"
#include "tnn_api.h"
#include "common.h"
#include <ipex.h>
#include "common_test.h"

using namespace sycl;
using json = nlohmann::json;

// Check if the output tensor is equal to the input tensor
bool checkTensorClose(const torch::Tensor &input, const torch::Tensor &output) {
    return torch::allclose(input, output, 1e-3);
}



template <typename T, typename T_ref, int WIDTH = 64>
void test_network_backward(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                           const int batch_size, Activation activation, Activation output_activation,
                           mlp_cpp::WeightInitMode weight_init_mode) {

    // main functionalities of backward and forward are tested in doctest_swifnet
    // here, we test only if the combination of encoding (tested in doctest_encoding) and swifnet works
    const T_ref input_val = 1.0;
    const T_ref target_val = 0.1;
    constexpr int padded_output_width = WIDTH;
    constexpr int padded_input_width = WIDTH;

    std::vector<T_ref> input_ref = test::common::create_padded_vector<T_ref>(input_width, input_val, padded_input_width);
    std::vector<T_ref> target_ref = test::common::create_padded_vector<T_ref>(output_width, target_val, padded_output_width);

    mlp_cpp::MLP<T_ref> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation,
                            output_activation, weight_init_mode);

    tnn::NetworkModule<T, WIDTH> Net(input_width, output_width, n_hidden_layers, activation,
                                     output_activation, false);

    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<T_ref, T>(mlp.getUnpackedWeights());
    auto network_torch_params = tnn::Module::convertVectorToTensor<T>(unpacked_weights).to(c10::ScalarType::BFloat16);
    torch::Tensor torch_params = network_torch_params;
    auto params = Net.initialize_params(torch_params);

    torch::Tensor input_tensor = torch::zeros({batch_size, WIDTH}, c10::ScalarType::BFloat16).to(torch::kXPU);
    // Fill the first 'input_width' columns with 'input_val'
    for (int i = 0; i < input_width; ++i) {
        input_tensor.index({at::indexing::Slice(), i}) = input_val;
    }

    auto output_net = Net.forward_pass(input_tensor);

    std::vector<T> output_vec = tnn::Module::convertTensorToVector<T>(output_net);

    auto fwd_result_ref = mlp.forward(input_ref, false);

    std::vector<T_ref> forw_ref = mlp_cpp::repeat_inner_vectors<T_ref>(fwd_result_ref, batch_size);

    if (!areVectorsWithinTolerance(output_vec, forw_ref, 2.0e-2)) {
        printVector("output_vec", output_vec);
        printVector("forw_ref", forw_ref);
    }
    CHECK(areVectorsWithinTolerance(output_vec, forw_ref, 2.0e-2));

    std::vector<mlp_cpp::Matrix<T_ref>> grad_matrices_ref(n_hidden_layers + 1, mlp_cpp::Matrix<T_ref>(1, 1));
    std::vector<std::vector<T_ref>> loss_grads_ref;
    std::vector<T_ref> loss_ref;

    std::vector<T_ref> dL_doutput_ref;
    std::vector<T_ref> dL_dinput_ref;

    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 1.0);

    DeviceMatrix<T> dL_doutput(batch_size, WIDTH, q);
    dL_doutput.fill(0.0).wait();
    std::vector<double> stacked_dL_doutput_ref = mlp_cpp::stack_vector(dL_doutput_ref, batch_size);

    dL_doutput.copy_from_host(mlp_cpp::convert_vector<T_ref, T>(stacked_dL_doutput_ref)).wait();

    auto [dL_dinput, net_grad] =
        Net.backward_pass(tnn::Module::convertDeviceMatrixToTorchTensor<T>(dL_doutput), false, true);
    auto net_grad_from_torch = tnn::Module::convertTensorToVector<T>(net_grad);

    std::vector<T> dL_dinput_vec = tnn::Module::convertTensorToVector<T>(dL_dinput);

    // flatten reference grad matrices
    std::vector<double> grads_ref;
    for (const auto &matrix : grad_matrices_ref) {
        for (size_t row = 0; row < matrix.rows(); ++row) {
            for (size_t col = 0; col < matrix.cols(); ++col) {
                grads_ref.push_back(matrix.data[row][col]);
            }
        }
    }

    std::vector<T> network_params_grad_vec = Net.get_network_grads();
    bool grads_within_tolerance = areVectorsWithinTolerance(network_params_grad_vec, grads_ref, 5.0e-2);
    if (!grads_within_tolerance) {
        printVector("grads_ref", grads_ref);
        printVector("network_params_grad_vec", network_params_grad_vec);
    }
    CHECK(areVectorsWithinTolerance(network_params_grad_vec, grads_ref, 5.0e-2));

    auto dL_dinput_ref_stacked = mlp_cpp::stack_vector(dL_dinput_ref, batch_size);

    if (!areVectorsWithinTolerance(dL_dinput_vec, dL_dinput_ref_stacked, 1.0e-2)) {
        printVector("dL_dinput_ref_stacked: ", dL_dinput_ref_stacked);
        printVector("dL_dinput_vec: ", dL_dinput_vec);
    }
    CHECK(areVectorsWithinTolerance(dL_dinput_vec, dL_dinput_ref_stacked, 1.0e-2));
}

TEST_CASE("Network - test bwd unpadded") {
   
    // test_network_backward<bf16, double, 16>(q, 16, 16, n_hidden_layers, 8, "sigmoid", "linear", "constant");
    auto test_function = [=](sycl::queue &q, const int batch_size, const int width, Activation activation,
                             Activation output_activation, mlp_cpp::WeightInitMode weight_init_mode, bool random_init) {
        typedef bf16 T;
        constexpr int n_hidden_layers = 1;
        if (width == 16) {
            // Define the parameters for creating IdentityEncoding
            test_network_backward<T, double, 16>(q, 16, 16, n_hidden_layers, batch_size, activation, output_activation,
                                                 weight_init_mode);
        } else if (width == 32) {
            // Define the parameters for creating IdentityEncoding
            test_network_backward<T, double, 32>(q, 32, 32, n_hidden_layers, batch_size, activation, output_activation,
                                                 weight_init_mode);
        } else if (width == 64) {
            // Define the parameters for creating IdentityEncoding
            test_network_backward<T, double, 64>(q, 64, 64, n_hidden_layers, batch_size, activation, output_activation,
                                                 weight_init_mode);
        } else if (width == 128) {
            // Define the parameters for creating IdentityEncoding
            test_network_backward<T, double, 128>(q, 128, 128, n_hidden_layers, batch_size, activation,
                                                  output_activation, weight_init_mode);
        } else
            throw std::invalid_argument("Unsupported width");
    };

    sycl::queue q(sycl::gpu_selector_v);

    std::vector<int> batch_sizes{8, 16, 32, 64};
    std::vector<int> widths{16, 32, 64, 128};
    std::vector<Activation> activations{Activation::None, Activation::ReLU, Activation::Sigmoid};
    std::vector<Activation> output_activations{Activation::None, Activation::Sigmoid};
    std::vector<mlp_cpp::WeightInitMode> weight_init_modes{mlp_cpp::WeightInitMode::random};
    std::vector<bool> random_inits{false};

    test::common::LoopOverParams(q, batch_sizes, widths, activations, output_activations, weight_init_modes, random_inits, test_function);
}

// TEST_CASE("Network - test bwd input padded") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 1;
//     const int input_dim = 8;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef bf16 T;
//         if (width == 16) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 16>(q, input_dim, 16, n_hidden_layers, batch_size, activation,
//                                                  output_activation, weight_init_mode);
//         } else if (width == 32) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 32>(q, input_dim, 32, n_hidden_layers, batch_size, activation,
//                                                  output_activation, weight_init_mode);
//         } else if (width == 64) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 64>(q, input_dim, 64, n_hidden_layers, batch_size, activation,
//                                                  output_activation, weight_init_mode);
//         } else if (width == 128) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 128>(q, input_dim, 128, n_hidden_layers, batch_size, activation,
//                                                   output_activation, weight_init_mode);
//         } else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64};
//     std::string activations[] = {"linear", "relu", "sigmoid"};
//     std::string output_activations[] = {"linear"};
//     std::string weight_init_modes[] = {"constant", "random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {

//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing grad WIDTH " + std::to_string(width) + " - activation: " + activation + " -
//                             output_activation: " + output_activation+ " - weight_init_mode: " + weight_init_mode + "
//                             - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation,
//                                 weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST_CASE("Network - test bwd output padded") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 1;
//     const int output_dim = 8;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef bf16 T;
//         if (width == 16) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 16>(q, 16, output_dim, n_hidden_layers, batch_size, activation,
//                                                  output_activation, weight_init_mode);
//         } else if (width == 32) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 32>(q, 32, output_dim, n_hidden_layers, batch_size, activation,
//                                                  output_activation, weight_init_mode);
//         } else if (width == 64) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 64>(q, 64, output_dim, n_hidden_layers, batch_size, activation,
//                                                  output_activation, weight_init_mode);
//         } else if (width == 128) {
//             // Define the parameters for creating IdentityEncoding
//             test_network_backward<T, double, 128>(q, 128, output_dim, n_hidden_layers, batch_size, activation,
//                                                   output_activation, weight_init_mode);
//         } else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64};
//     std::string activations[] = {"linear", "relu", "sigmoid"};
//     std::string output_activations[] = {"linear"};
//     std::string weight_init_modes[] = {"constant", "random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing grad WIDTH " + std::to_string(width) + " - activation: " + activation + " -
//                             output_activation: " + output_activation+ " - weight_init_mode: " + weight_init_mode + "
//                             - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation,
//                                 weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
