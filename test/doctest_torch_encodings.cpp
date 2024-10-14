/**
 * @file doctest_torch_encodings.cpp
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
#include "result_check.h"
#include "tnn_api.h"
#include <ipex.h>

// Check if the output tensor is equal to the input tensor
bool checkTensorClose(const torch::Tensor &input, const torch::Tensor &output) {
    return torch::allclose(input, output, 1e-3);
}

template <typename T>
void test_encoding_training_from_loaded_file(const int batch_size, const int input_width, const int output_width,
                                             std::string filepath, sycl::queue &q) {
    float epsilon = 1e-2;
    DeviceMatrix<float> input(batch_size, input_width, q);
    input.fill(0.0f).wait();

    DeviceMatrix<T> output(batch_size, output_width, q);
    output.fill(0.0f).wait();
    json encoding_config = io::loadJsonConfig(filepath + "/encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;

    std::vector<T> params = io::loadVectorFromCSV<T>(filepath + "encoding_params.csv");
    std::vector<float> input_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    std::vector<T> output_ref = io::loadVectorFromCSV<T>(filepath + "output_encoding.csv");

    // Initialize EncodingModule with Identity encoding
    tnn::EncodingModule<T> encoding(encoding_config);
    auto torch_params_converted = tnn::Module::convertVectorToTensor(params);
    auto torch_params = encoding.initialize_params(torch_params_converted);

    CHECK(areVectorsWithinTolerance(tnn::Module::convertTensorToVector<T>(torch_params), params, epsilon));

    input.copy_from_host(input_ref).wait();

    auto input_encoding = tnn::Module::convertDeviceMatrixToTorchTensor(input);
    auto output_encoding = encoding.forward_pass(input_encoding);
    std::vector<T> output_vec = tnn::Module::convertTensorToVector<T>(output_encoding);

    // Check if the actual vector is equal to the expected vector within the tolerance
    CHECK(areVectorsWithinTolerance(output_vec, output_ref, epsilon));

    // testing backward now
    std::vector<T> params_grad_ref = io::loadVectorFromCSV<T>(filepath + "params_grad.csv");
    std::vector<T> dL_doutput_ref = io::loadVectorFromCSV<T>(filepath + "dL_doutput.csv");

    DeviceMatrix<T> dL_doutput(batch_size, output_width, q);
    dL_doutput.fill(0.0f).wait();
    dL_doutput.copy_from_host(dL_doutput_ref).wait();

    CHECK(isVectorWithinTolerance(encoding.get_encoding_grads(), 0.0f, epsilon)); // default init to 0.0f

    auto [dL_dinput_empty, torch_grad] =
        encoding.backward_pass(tnn::Module::convertDeviceMatrixToTorchTensor<T>(dL_doutput), input_encoding);
    CHECK(dL_dinput_empty.numel() == 0);

    std::vector<float> enc_params = encoding.get_encoding_params();
    std::vector<float> enc_grads = encoding.get_encoding_grads();
    CHECK(areVectorsWithinTolerance(tnn::Module::convertTensorToVector<T>(torch_grad), enc_grads, epsilon));
    CHECK(areVectorsWithinTolerance(enc_params, params, 1e-5));

    CHECK(areVectorsWithinTolerance(enc_grads, params_grad_ref, epsilon));
}

template <typename T>
void test_encoding_with_no_params_forward_from_loaded_file(const int batch_size, const int input_width,
                                                           const int output_width, std::string filepath,
                                                           sycl::queue &q) {
    DeviceMatrix<float> input(batch_size, input_width, q);
    input.fill(0.0f).wait();

    json encoding_config = io::loadJsonConfig(filepath + "/encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;

    // Initialize EncodingModule with Identity encoding
    tnn::EncodingModule<float> encoding_module(encoding_config);

    std::vector<float> input_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    std::vector<T> output_ref = io::loadVectorFromCSV<T>(filepath + "output_encoding.csv");

    input.copy_from_host(input_ref).wait();

    auto output_tensor = encoding_module.forward_pass(tnn::Module::convertDeviceMatrixToTorchTensor(input));

    auto output_vec = tnn::Module::convertTensorToVector<T>(output_tensor);
    CHECK(input_ref.size() == input.size());
    CHECK(output_ref.size() == output_vec.size());

    // Check if the actual vector is equal to the expected vector within the tolerance
    CHECK(areVectorsWithinTolerance(output_vec, output_ref, 1.0e-2));
}

TEST_CASE("EncodingModule::Identity") {
    sycl::queue q;
    int input_width = 3;
    // Example JSON config for Identity encoding
    json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                         {EncodingParams::SCALE, 1.0},
                         {EncodingParams::OFFSET, 0.0},
                         {EncodingParams::ENCODING, EncodingNames::IDENTITY}};

    // Initialize EncodingModule with Identity encoding
    tnn::EncodingModule<float> encoding_module(encoding_config);

    SUBCASE("forward_pass with Identity encoding") {
        const int batch_size = 2;
        auto input_tensor = torch::randn({batch_size, input_width}).to(torch::kXPU);

        // Call forward_pass (use_inference flag has been set to false as an example)
        auto output_tensor = encoding_module.forward_pass(input_tensor);

        // Check if input and output tensors are the same
        CHECK(checkTensorClose(input_tensor, output_tensor));
    }

    SUBCASE("initialize_params for Identity encoding") {
        // Initialize parameters to some predefined values
        auto initialized_params = encoding_module.initialize_params();

        // In the Identity encoding, there should be no learned parameters
        CHECK(initialized_params.numel() == 0);
    }
    SUBCASE("Not padded") {
        const int batch_size = 2;
        const int output_width = 3;

        auto input_tensor = torch::full({batch_size, input_width}, 1.23).to(torch::kXPU);
        auto output_tensor = encoding_module.forward_pass(input_tensor);

        CHECK(checkTensorClose(input_tensor, output_tensor));
    }

// Use conditional preprocessor check to include tests that depend on an external path
#ifdef TEST_PATH
    SUBCASE("Check results loaded float") {
        const int batch_size = 256;
        const int output_width = 3;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/identity/";
        test_encoding_with_no_params_forward_from_loaded_file<float>(batch_size, input_width, output_width, filepath,
                                                                     q);
    }

    SUBCASE("Check results loaded bf16") {
        const int batch_size = 256;
        const int output_width = 3;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/identity/";
        test_encoding_with_no_params_forward_from_loaded_file<bf16>(batch_size, input_width, output_width, filepath, q);
    }
#endif
}

TEST_CASE("EncodingModule::Spherical Harmonics") {
    sycl::queue q;

    SUBCASE("Not padded") {
        const int batch_size = 1;
        const int input_width = 3;
        const int DEGREE = input_width;

        sycl::queue q;
        auto input_tensor = torch::ones({batch_size, input_width}).to(torch::kXPU);

        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::DEGREE, DEGREE},
                                   {EncodingParams::ENCODING, EncodingNames::SPHERICALHARMONICS}};
        tnn::EncodingModule<float> encoding_module(encoding_config);

        auto output_tensor = encoding_module.forward_pass(input_tensor);
        auto output_vec = tnn::Module::convertTensorToVector<float>(output_tensor);

        const std::vector<float> reference_out = {0.2821,  -0.4886, 0.4886,  -0.4886, 1.0925,
                                                  -1.0925, 0.6308,  -1.0925, 0.0000};

        const double epsilon = 1e-2;
        // Check if the actual vector is equal to the expected vector within the tolerance
        CHECK(areVectorsWithinTolerance(output_vec, reference_out, epsilon));
    }

#ifdef TEST_PATH
    SUBCASE("Check results loaded float") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 256;
        const int output_width = 16;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/spherical/";
        test_encoding_with_no_params_forward_from_loaded_file<float>(batch_size, input_width, output_width, filepath,
                                                                     q);
    }
#endif
}

TEST_CASE("EncodingModule::Grid Encoding") {
    SUBCASE("Test grid encoding using create_grid_encoding instead of factory") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 1;
        const int padded_output_width = 32;
        sycl::queue q;

        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(1.0f).wait();

        DeviceMatrix<float> output_float(batch_size, padded_output_width, q);
        output_float.fill(1.23f).wait(); // fill with something to check if it is written to

        const json encoding_config{
            {EncodingParams::N_DIMS_TO_ENCODE, input_width}, {EncodingParams::ENCODING, EncodingNames::GRID},
            {EncodingParams::GRID_TYPE, GridType::Hash},     {EncodingParams::N_LEVELS, 16},
            {EncodingParams::N_FEATURES_PER_LEVEL, 2},       {EncodingParams::LOG2_HASHMAP_SIZE, 19},
            {EncodingParams::BASE_RESOLUTION, 16},           {EncodingParams::PER_LEVEL_SCALE, 2.0}};

        tnn::EncodingModule<float> encoding_module(encoding_config);

        auto output_tensor = encoding_module.forward_pass(tnn::Module::convertDeviceMatrixToTorchTensor(input));

        const std::vector<float> reference_out = {1.1325,  -4.1187, 1.0490,  -4.0174, 0.5364, 0.7808, -3.2306, 3.6359,
                                                  1.1146,  3.4690,  -1.1802, 0.8106,  0.0119, 1.1921, -2.4438, -2.1577,
                                                  -2.4140, 2.0504,  -2.9564, -1.2577, 0.9000, 3.6776, 3.1948,  -1.4663,
                                                  -4.4584, -2.0742, -0.9418, -0.6199, 4.9114, 1.2636, 3.4571,  -9.1076};
        auto output_vec = tnn::Module::convertTensorToVector<float>(output_tensor / 1e-05);

        // Check if the actual vector is equal to the expected vector within the tolerance
        CHECK(areVectorsWithinTolerance(output_vec, reference_out, 1.0e-2));
    }

#ifdef TEST_PATH

    SUBCASE("Check results loaded, base resolution 15") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 256;
        const int output_width = 32;
        sycl::queue q;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/gridbase_resolution_15/";
        test_encoding_training_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
    }
    SUBCASE("Check results loaded, n_levels 15") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 256;
        const int output_width = 30;
        sycl::queue q;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/gridn_levels15/";

        test_encoding_training_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
    }
    SUBCASE("Check results loaded, per level scale 1") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 256;
        const int output_width = 32;
        sycl::queue q;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/gridper_level_scale1/";

        test_encoding_training_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
    }
    SUBCASE("Check results loaded, per level scale 1.5") {
        // SWIFTNET
        const int input_width = 2;
        const int batch_size = 256;
        const int output_width = 32;
        sycl::queue q;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/gridper_level_scale15/";

        test_encoding_training_from_loaded_file<float>(batch_size, input_width, output_width, filepath, q);
    }
#endif
}