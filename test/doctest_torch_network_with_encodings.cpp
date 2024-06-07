/**
 * @file doctest_torch_network_with_encodings.cpp
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
#include <ipex.h>
// Check if the output tensor is equal to the input tensor
bool checkTensorClose(const torch::Tensor &input, const torch::Tensor &output) {
    return torch::allclose(input, output, 1e-3);
}

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;
using json = nlohmann::json;

template <typename T_enc, typename T_net, int WIDTH = 64>
void test_network_with_encoding_backward(sycl::queue &q, const int input_width, const int output_width,
                                         const int n_hidden_layers, const int batch_size, std::string activation,
                                         std::string weight_init_mode, const json &encoding_config) {
    // main functionalities of backward and forward are tested in doctest_swifnet
    // here, we test only if the combination of encoding (tested in doctest_encoding) and swifnet works
    const float input_val = 1.0f;
    const float target_val = 0.1;
    std::vector<float> input_ref(input_width, input_val);
    std::vector<float> target_ref(output_width, target_val);
    CHECK(input_width == output_width);
    // mlp reference (batch size = 1) assumes this. if this is changed, ensure checks are still correct
    CHECK(input_width == WIDTH);
    Activation network_activation;
    if (activation == "relu") {
        network_activation = Activation::ReLU;
    } else if (activation == "linear") {
        network_activation = Activation::None;
    }

    mlp_cpp::MLP<float> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation, "linear",
                            weight_init_mode);

    tnn::NetworkWithEncodingModule<T_enc, T_net, WIDTH> Net(input_width, output_width, n_hidden_layers,
                                                            network_activation, Activation::None, encoding_config);

    std::vector<T_net> unpacked_weights = mlp_cpp::convert_vector<float, T_net>(mlp.getUnpackedWeights());
    auto network_torch_params =
        tnn::Module::convertVectorToTensor<T_net>(
            io::get_packed_weights<T_net, WIDTH>(unpacked_weights, n_hidden_layers, input_width, WIDTH))
            .to(torch::kFloat32);
    torch::Tensor torch_params = network_torch_params;
    Net.initialize_params(torch_params);
    auto output_net = Net.forward_pass(torch::ones({batch_size, input_width}).to(torch::kXPU) * input_val);
    std::vector<T_net> output_vec = tnn::Module::convertTensorToVector<T_net>(output_net);

    auto fwd_result_ref = mlp.forward(input_ref, false);
    std::vector<float> forw_ref = mlp_cpp::repeat_inner_vectors<float>(fwd_result_ref, batch_size);

    std::vector<mlp_cpp::Matrix<float>> grad_matrices_ref(n_hidden_layers + 1, mlp_cpp::Matrix<float>(1, 1));
    std::vector<std::vector<float>> loss_grads_ref;
    std::vector<float> loss_ref;

    std::vector<float> dL_doutput_ref;
    std::vector<float> dL_dinput_ref;

    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 1.0);

    DeviceMatrix<T_net> dL_doutput(batch_size, WIDTH, q);
    dL_doutput.fill(0.0).wait();

    std::vector<float> stacked_loss_grads_back_ref = mlp_cpp::stack_vector(loss_grads_ref.back(), batch_size);
    dL_doutput.copy_from_host(mlp_cpp::convert_vector<float, T_net>(stacked_loss_grads_back_ref)).wait();

    auto [dL_dinput_empty, net_grad] =
        Net.backward_pass(tnn::Module::convertDeviceMatrixToTorchTensor<T_net>(dL_doutput), false, false);
    CHECK(dL_dinput_empty.numel() == 0);

    auto net_grad_from_torch = tnn::Module::convertTensorToVector<T_net>(net_grad);

    // flatten reference grad matrices
    std::vector<double> grads_ref;
    for (const auto &matrix : grad_matrices_ref) {
        for (size_t row = 0; row < matrix.rows(); ++row) {
            for (size_t col = 0; col < matrix.cols(); ++col) {
                grads_ref.push_back(matrix.data[row][col]);
            }
        }
    }
    std::vector<T_net> network_params_grad_vec = Net.get_network_grads();
    bool grads_within_tolerance = areVectorsWithinTolerance(network_params_grad_vec, grads_ref, 1.0e-2);
    if (!grads_within_tolerance) {
        printVector("grads_ref", grads_ref);
        printVector("network_params_grad_vec", network_params_grad_vec);
    }
    CHECK(areVectorsWithinTolerance(network_params_grad_vec, grads_ref, 1.0e-2));
}

template <typename T_enc, typename T_net, int WIDTH = 64>
void test_network_with_encoding_inference_loaded(sycl::queue &q, std::string filepath, const int n_hidden_layers,
                                                 const int batch_size, const int unpadded_output_width,
                                                 const int encoding_input_width) {
    json encoding_config = io::loadJsonConfig(filepath + "encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = encoding_input_width;
    tnn::NetworkWithEncodingModule<T_enc, T_net, WIDTH> Net(encoding_input_width, unpadded_output_width,
                                                            n_hidden_layers, Activation::ReLU, Activation::None,
                                                            encoding_config);

    std::vector<T_enc> encoding_params = io::loadVectorFromCSV<T_enc>(filepath + "encoding_params.csv");
    std::vector<T_net> network_params = io::load_weights_as_packed_from_file<T_net, WIDTH>(
        filepath + "network_params.csv", n_hidden_layers, WIDTH, WIDTH);
    auto network_torch_params = tnn::Module::convertVectorToTensor<T_net>(network_params).to(torch::kFloat32);

    torch::Tensor torch_params = network_torch_params;
    if (encoding_params.size()) {
        auto encoding_torch_params = tnn::Module::convertVectorToTensor<T_enc>(encoding_params).to(torch::kFloat32);
        torch_params = torch::cat({network_torch_params, encoding_torch_params}, 0);
    }

    Net.initialize_params(torch_params);

    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    std::vector<float> input_encoding_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    input_encoding.copy_from_host(input_encoding_ref);
    q.wait();
    auto input_torch = tnn::Module::convertDeviceMatrixToTorchTensor(input_encoding);

    auto output_net = Net.forward_pass(input_torch);
    auto output_inference_net = Net.inference(input_torch);

    checkTensorClose(output_inference_net, output_net);

    std::vector<T_net> output_vec = tnn::Module::convertTensorToVector<T_net>(output_net);

    std::vector<T_net> network_output_ref = io::loadVectorFromCSV<T_net>(filepath + "output_network.csv");
    if (!areVectorsWithinTolerance(output_vec, network_output_ref, 1.0e-2)) {
        printVector("output_vec", output_vec);
        printVector("network_output_ref", network_output_ref);
    }
    CHECK(areVectorsWithinTolerance(output_vec, network_output_ref, 1.0e-2));
}

template <typename T_enc, typename T_net, int WIDTH = 64>
void test_network_with_encoding_training_loaded(sycl::queue &q, std::string filepath, const int n_hidden_layers,
                                                const int batch_size, const int unpadded_output_width,
                                                const int encoding_input_width) {
    float epsilon = 1e-2;
    json encoding_config = io::loadJsonConfig(filepath + "encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = encoding_input_width;

    tnn::NetworkWithEncodingModule<T_enc, T_net, WIDTH> Net(encoding_input_width, unpadded_output_width,
                                                            n_hidden_layers, Activation::ReLU, Activation::None,
                                                            encoding_config);

    std::vector<T_enc> encoding_params_ref = io::loadVectorFromCSV<T_enc>(filepath + "encoding_params.csv");
    std::vector<T_net> network_params = io::load_weights_as_packed_from_file<T_net, WIDTH>(
        filepath + "network_params.csv", n_hidden_layers,
        Net.get_network_with_encoding()->get_network()->get_input_width(),
        Net.get_network_with_encoding()->get_network()->get_output_width());

    auto network_torch_params = tnn::Module::convertVectorToTensor<T_net>(network_params).to(torch::kFloat32);
    torch::Tensor torch_params = network_torch_params;
    if (encoding_params_ref.size()) {
        auto encoding_torch_params = tnn::Module::convertVectorToTensor<T_enc>(encoding_params_ref).to(torch::kFloat32);
        torch_params = torch::cat({network_torch_params, encoding_torch_params}, 0);
    }

    auto init_params = Net.initialize_params(torch_params);

    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    std::vector<float> input_encoding_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");

    input_encoding.copy_from_host(input_encoding_ref);
    q.wait();
    auto input_encoding_torch = tnn::Module::convertDeviceMatrixToTorchTensor(input_encoding);
    auto output_net = Net.forward_pass(input_encoding_torch);

    std::vector<T_net> output_vec = tnn::Module::convertTensorToVector<T_net>(output_net);

    std::vector<T_net> network_output_ref = io::loadVectorFromCSV<T_net>(filepath + "output_network.csv");
    if (!areVectorsWithinTolerance(output_vec, network_output_ref, 1.0e-2)) {
        printVector("output_vec", output_vec);
        printVector("network_output_ref", network_output_ref);
    }
    CHECK(areVectorsWithinTolerance(output_vec, network_output_ref, 1.0e-2));

    DeviceMatrix<T_net> dL_doutput(batch_size, unpadded_output_width, q);
    dL_doutput.fill(0.0f).wait();

    std::vector<T_net> dL_doutput_ref = io::loadVectorFromCSV<T_net>(filepath + "dL_doutput.csv");
    dL_doutput.copy_from_host(dL_doutput_ref).wait();

    torch::Tensor dL_doutput_torch = tnn::Module::convertDeviceMatrixToTorchTensor<T_net>(dL_doutput);
    // we need to pad grad_output to network_output_width first
    int64_t pad_right = Net.n_output_dims() - unpadded_output_width;

    if (pad_right > 0) {
        // Applying padding, only on the right side (column wise)
        torch::nn::functional::PadFuncOptions pad_options({0, pad_right, 0, 0});
        dL_doutput_torch = torch::nn::functional::pad(dL_doutput_torch, pad_options);
    }

    torch::Tensor input_grad_tensor, net_grad_tensor;
    std::tuple<torch::Tensor, torch::Tensor> grad_tensors;

    if (encoding_params_ref.size()) {
        grad_tensors = Net.backward_pass(dL_doutput_torch, input_encoding_torch, false, false);
    } else {
        grad_tensors = Net.backward_pass(dL_doutput_torch, false, false);
    }
    std::tie(input_grad_tensor, net_grad_tensor) = grad_tensors;

    CHECK(input_grad_tensor.numel() == 0); // Currently, we don't support input_grads, thus it should be empty

    // network param grad
    std::vector<T_net> network_params_grad_vec = Net.get_network_grads();
    // First N elements
    torch::Tensor network_grad_from_bwd =
        net_grad_tensor.slice(/*dim=*/0, /*start=*/0, /*end=*/network_params_grad_vec.size());

    if (encoding_params_ref.size()) {
        // If there is encoding_params, then all_gradients is T_enc
        std::vector<T_enc> network_grad_from_bwd_vec = tnn::Module::convertTensorToVector<T_enc>(network_grad_from_bwd);
        CHECK(areVectorsWithinTolerance(network_params_grad_vec, network_grad_from_bwd_vec, epsilon));
    } else {
        // If there is no encoding_params, then all_gradients is T_net (just Swiftnet backward)
        std::vector<T_net> network_grad_from_bwd_vec = tnn::Module::convertTensorToVector<T_net>(network_grad_from_bwd);
        CHECK(areVectorsWithinTolerance(network_params_grad_vec, network_grad_from_bwd_vec, epsilon));
    }

    std::vector<T_net> network_params_grad_ref = io::loadVectorFromCSV<T_net>(filepath + "network_params_grad.csv");

    if (!areVectorsWithinTolerance(network_params_grad_vec, network_params_grad_ref, epsilon)) {
        double sum_vec = 0.0;
        double sum_ref = 0.0;
        for (auto &val : network_params_grad_ref) {
            sum_ref += val;
        }
        for (auto &val : network_params_grad_vec) {
            sum_vec += val;
        }
        printVector("network_params_grad_vec", network_params_grad_vec);
        printVector("network_params_grad_ref", network_params_grad_ref);
        std::cout << "network_params_grad_ref sum: " << sum_ref << std::endl;
        std::cout << "network_params_grad_vec sum: " << sum_vec << std::endl;
    }
    CHECK(areVectorsWithinTolerance(network_params_grad_vec, network_params_grad_ref, epsilon));

    if (encoding_config[EncodingParams::ENCODING] == EncodingNames::GRID) {
        // encoding param grad
        CHECK(areVectorsWithinTolerance(Net.get_encoding_params(), encoding_params_ref, epsilon));

        // Elements from N to the end of the tensor
        torch::Tensor encoding_grad_from_bwd =
            net_grad_tensor.slice(/*dim=*/0, /*start=*/network_params_grad_vec.size());

        std::vector<T_enc> encoding_grad_from_bwd_vec =
            tnn::Module::convertTensorToVector<T_enc>(encoding_grad_from_bwd);

        // CHECK params after backward pass it shouldn't have changed because no optimise step
        std::vector<T_enc> encoding_params_grad_ref =
            io::loadVectorFromCSV<T_enc>(filepath + "encoding_params_grad.csv");
        std::vector<float> encoding_params_grad_vec = Net.get_encoding_grads();

        // ensure that what comes out from bwd (all_gradients which combines enc and net gradients) is the same as if we
        // use get_encoding_grads function
        CHECK(areVectorsWithinTolerance(encoding_params_grad_vec, encoding_grad_from_bwd_vec, epsilon));

        if (!areVectorsWithinTolerance(encoding_params_grad_vec, encoding_params_grad_ref, epsilon)) {
            double sum_vec = 0.0;
            double sum_ref = 0.0;
            for (auto &val : encoding_params_grad_ref) {
                sum_ref += val;
            }
            for (auto &val : encoding_params_grad_vec) {
                sum_vec += val;
            }
            std::cout << "encoding_params_grad_ref sum: " << sum_ref << std::endl;
            std::cout << "encoding_params_grad_ref sum: " << sum_vec << std::endl;
        }
        CHECK(areVectorsWithinTolerance(encoding_params_grad_vec, encoding_params_grad_ref, epsilon));
    }
}

TEST_CASE("tinydpcppnn::network_with_encoding step-by-step") {
    sycl::queue q(gpu_selector_v);

#ifdef TEST_PATH
    SUBCASE("Grid encoding inference, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_hashgrid/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 2;
        const int unpadded_output_width = 64;
        test_network_with_encoding_inference_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                     unpadded_output_width, encoding_input_width);
    }

    SUBCASE("Identity encoding inference, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 16;
        const int unpadded_output_width = 64;
        test_network_with_encoding_inference_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                     unpadded_output_width, encoding_input_width);
    }

    SUBCASE("Grid encoding training, constant weights, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_hashgrid_constant/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 2;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }

    SUBCASE("Grid encoding training, random weights, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_hashgrid_random/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 2;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }
    SUBCASE("Identity encoding training, linspace weights, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) +
            "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity_linspace_paddedFalse/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 64;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }
    SUBCASE("Identity encoding training, constant weights, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) +
            "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity_constant_paddedFalse/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 64;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }
    SUBCASE("Identity encoding training, random weights, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) +
            "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity_random_paddedFalse/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 64;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }

    SUBCASE("Identity encoding training, linspace weights, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) +
            "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity_linspace_paddedTrue/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 16;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }
    SUBCASE("Identity encoding training, constant weights, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) +
            "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity_constant_paddedTrue/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 16;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }
    SUBCASE("Identity encoding training, random weights, loaded data") {
        std::string filepath = std::string(TEST_PATH) +
                               "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity_random_paddedTrue/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 16;
        const int unpadded_output_width = 64;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }

    SUBCASE("Identity encoding training, linspace weights, padded output, loaded data") {
        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/"
                                                        "nwe_identity_linspace_paddedFalse_output_paddedTrue/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 64;
        const int unpadded_output_width = 5;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }
    SUBCASE("Identity encoding training, constant weights, padded output, loaded data") {
        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/"
                                                        "nwe_identity_constant_paddedFalse_output_paddedTrue/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 64;
        const int unpadded_output_width = 5;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }
    SUBCASE("Identity encoding training, random weights, padded output, loaded data") {
        std::string filepath =
            std::string(TEST_PATH) +
            "/tiny-dpcpp-data/ref_values/network_with_grid_encoding/nwe_identity_random_paddedFalse_output_paddedTrue/";
        const int n_hidden_layers = 2;
        const int batch_size = 256;
        const int encoding_input_width = 64;
        const int unpadded_output_width = 5;
        test_network_with_encoding_training_loaded<float, bf16, 64>(q, filepath, n_hidden_layers, batch_size,
                                                                    unpadded_output_width, encoding_input_width);
    }

#endif
}

TEST_CASE("Network with Identity Encoding - test bwd unpadded") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;
    auto test_function = [=](auto T_type, sycl::queue &q, const int width, const int batch_size, std::string activation,
                             std::string weight_init_mode) {
        using T = decltype(T_type);
        if (width == 16) {
            // Define the parameters for creating IdentityEncoding
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<float, T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation,
                                                              weight_init_mode, encoding_config);
        } else if (width == 32) {
            // Define the parameters for creating IdentityEncoding
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<float, T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation,
                                                              weight_init_mode, encoding_config);
        } else if (width == 64) {
            // Define the parameters for creating IdentityEncoding
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<float, T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation,
                                                              weight_init_mode, encoding_config);
        } else if (width == 128) {
            // Define the parameters for creating IdentityEncoding
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<float, T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation,
                                                               weight_init_mode, encoding_config);
        } else
            throw std::invalid_argument("Unsupported width");
    };
    const int widths[] = {16, 32, 64, 128};
    const int batch_sizes[] = {8, 16, 32, 64};
    std::string activations[] = {"linear", "relu"};
    std::string weight_init_modes[] = {"constant", "random"};

    auto bf16_type = sycl::ext::oneapi::bfloat16{};
    auto half_type = sycl::half{};

    std::array<decltype(bf16_type), 2> types = {bf16_type, half_type};
    for (auto type : types) {
        std::string type_name = (type == bf16_type) ? "bfloat16" : "half";

        for (int batch_size : batch_sizes) {
            for (int width : widths) {
                for (std::string activation : activations) {
                    for (std::string weight_init_mode : weight_init_modes) {
                        std::string testName = "Testing grad " + type_name + " WIDTH " + std::to_string(width) +
                                               " - activation: " + activation +
                                               " - weight_init_mode: " + weight_init_mode +
                                               " - Batch size: " + std::to_string(batch_size);
                        SUBCASE(testName.c_str()) {
                            CHECK_NOTHROW(test_function(type, q, width, batch_size, activation, weight_init_mode));
                        }
                    }
                }
            }
        }
    }
}
