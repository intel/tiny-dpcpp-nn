/**
 * @file doctest_encodings.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief File with test of the encodings.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <vector>

#include "doctest/doctest.h"
#include "encoding_factory.h"
#include "io.h"
#include "result_check.h"

using bf16 = sycl::ext::oneapi::bfloat16;
using tinydpcppnn::encodings::grid::GridEncoding;
using json = nlohmann::json;

template <typename T> void initialize_arange(std::vector<T> &vec) {

    // Repeat the col_vector and perform the operations
    double offset = (double)vec.size() / 2.0;

    for (long long i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<T>(i / offset - 1.0);
    }
}

template <typename T>
void test_encoding_with_no_params_forward_from_loaded_file(const int batch_size, const int input_width,
                                                           const int output_width, std::string filepath,
                                                           sycl::queue &q) {
    DeviceMatrix<float> input(batch_size, input_width, q);
    input.fill(0.0f).wait();

    DeviceMatrix<T> output(batch_size, output_width, q);
    output.fill(0.0f).wait();
    auto output_view = output.GetView();

    json config = io::loadJsonConfig(filepath + "/encoding_config.json");
    config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;

    std::shared_ptr<Encoding<T>> encoding = create_encoding<T>(config);
    encoding->set_padded_output_width(output_width);

    std::vector<T> params = io::loadVectorFromCSV<T>(filepath + "encoding_params.csv");
    CHECK(params.size() == 0);
    std::vector<float> input_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    std::vector<T> output_ref = io::loadVectorFromCSV<T>(filepath + "output_encoding.csv");

    CHECK(input_ref.size() == input.size());
    CHECK(output_ref.size() == output.size());

    input.copy_from_host(input_ref).wait();
    auto input_view = input.GetView();

    std::unique_ptr<Context> model_ctx = encoding->forward_impl(&q, input_view, &output_view);
    q.wait();

    // Check if the actual vector is equal to the expected vector within the tolerance
    CHECK(areVectorsWithinTolerance(output.copy_to_host(), output_ref, 1.0e-2));
}

template <typename T>
void test_encoding_training_from_loaded_file(const int batch_size, const int input_width, const int output_width,
                                             std::string filepath, sycl::queue &q) {
    DeviceMatrix<float> input(batch_size, input_width, q);
    input.fill(0.0f).wait();

    DeviceMatrix<T> output(batch_size, output_width, q);
    output.fill(0.0f).wait();
    auto output_view = output.GetView();

    json config = io::loadJsonConfig(filepath + "/encoding_config.json");
    config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;

    std::shared_ptr<Encoding<T>> encoding = create_encoding<T>(config, output_width, q);

    std::vector<T> params = io::loadVectorFromCSV<T>(filepath + "encoding_params.csv");
    std::vector<float> input_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    std::vector<T> output_ref = io::loadVectorFromCSV<T>(filepath + "output_encoding.csv");

    if (!encoding->get_n_params()) {
        throw std::runtime_error(
            "n_params is 0, DeviceMem cannot be created with size 0 and for training, we require params to be trained");
    }
    DeviceMatrix<T> gradients(encoding->get_n_params(), 1, q);
    gradients.fill(0.0).wait();
    auto gradients_view = gradients.GetView();


    // for grid encoding this is true
    encoding->get_params()->copy_from_host(params);

    CHECK(input_ref.size() == input.size());
    CHECK(output_ref.size() == output.size());

    input.copy_from_host(input_ref).wait();
    auto input_view = input.GetView();

    std::unique_ptr<Context> model_ctx = encoding->forward_impl(&q, input_view, &output_view);
    q.wait();

    // Check if the actual vector is equal to the expected vector within the tolerance
    CHECK(areVectorsWithinTolerance(output.copy_to_host(), output_ref, 1.0e-2));

    // testing backward now
    std::vector<T> params_grad_ref = io::loadVectorFromCSV<T>(filepath + "params_grad.csv");
    std::vector<T> dL_doutput_ref = io::loadVectorFromCSV<T>(filepath + "dL_doutput.csv");

    DeviceMatrix<T> dL_doutput(batch_size, output_width, q);
    dL_doutput.copy_from_host(dL_doutput_ref).wait();
    auto dL_doutput_view = dL_doutput.GetView();

    encoding->backward_impl(&q, *model_ctx, input_view, dL_doutput_view, &gradients_view);
    q.wait();

    std::vector<float> enc_params(encoding->get_n_params());
    std::vector<float> enc_grads = gradients.copy_to_host();

    q.memcpy(enc_params.data(), encoding->get_params(), encoding->get_n_params() * sizeof(float)).wait();
    // pure sanity check that params didn't change, we loaded them and set them

    float epsilon = 1e-2;
    CHECK(areVectorsWithinTolerance(enc_params, params, epsilon));

    CHECK(areVectorsWithinTolerance(enc_grads, params_grad_ref, epsilon));
}

TEST_CASE("tinydpcppnn::encoding Identity") {

    SUBCASE("Not padded") {
        const int batch_size = 2;
        const int input_width = 3;
        const int output_width = 3;

        sycl::queue q;
        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(1.23f).wait();

        DeviceMatrix<float> output_float(batch_size, output_width, q);
        output_float.fill(0.0f).wait();
        auto output_view = output_float.GetView();

        // Define the parameters for creating IdentityEncoding
        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::SCALE, 1.0},
                                   {EncodingParams::OFFSET, 0.0},
                                   {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
        std::shared_ptr<Encoding<float>> network = create_encoding<float>(encoding_config, q, output_width);
        std::unique_ptr<Context> model_ctx = network->forward_impl(input.GetView(), &output_view);
        q.wait();

        std::vector<float> in = input.copy_to_host();
        std::vector<float> out = output_float.copy_to_host();

        const float epsilon = 1e-3; // Set the tolerance for floating-point comparisons

        // Check if the actual vector is equal to the expected vector within the tolerance
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(static_cast<float>(in[i]) == doctest::Approx(out[i]).epsilon(epsilon));
        }
    }

#ifdef TEST_PATH
    SUBCASE("Check results loaded float") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 256;
        const int output_width = 3;
        sycl::queue q;
        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/identity/";
        test_encoding_with_no_params_forward_from_loaded_file<float>(batch_size, input_width, output_width, filepath,
                                                                     q);
    }

    SUBCASE("Check results loaded bf16") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 256;
        const int output_width = 3;
        sycl::queue q;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/identity/";
        test_encoding_with_no_params_forward_from_loaded_file<bf16>(batch_size, input_width, output_width, filepath, q);
    }
#endif
}

TEST_CASE("tinydpcppnn::encoding Spherical Harmonics") {

    SUBCASE("Not padded") {
        const int batch_size = 2;
        const int input_width = 3;
        const int output_width = 3;
        const int DEGREE = 1;

        sycl::queue q;
        std::vector<float> input_float(input_width * batch_size);
        initialize_arange(input_float);
        DeviceMatrix<float> input(batch_size, input_width, q);
        input.copy_from_host(input_float).wait();

        DeviceMatrix<float> output_float(batch_size, output_width, q);
        output_float.fill(0.0f).wait();
        auto output_view = output_float.GetView();
        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::DEGREE, DEGREE},
                                   {EncodingParams::ENCODING, EncodingNames::SPHERICALHARMONICS}};
        std::shared_ptr<Encoding<float>> network = create_encoding<float>(encoding_config, q, output_width);
        std::unique_ptr<Context> model_ctx = network->forward_impl(input.GetView(), &output_view);
        q.wait();

        std::vector<float> out = output_float.copy_to_host();
        const std::vector<float> reference_out = {0.2821, 1.0, 1.0, 0.2821, 1.0, 1.0};

        const double epsilon = 1e-3;
        // Check if the actual vector is equal to the expected vector within the tolerance
        CHECK(areVectorsWithinTolerance(out, reference_out, epsilon));
    }

#ifdef TEST_PATH
    SUBCASE("Check results loaded float") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 256;
        const int output_width = 16;
        sycl::queue q;

        std::string filepath = std::string(TEST_PATH) + "/tiny-dpcpp-data/ref_values/encoding/spherical/";
        test_encoding_with_no_params_forward_from_loaded_file<float>(batch_size, input_width, output_width, filepath,
                                                                     q);
    }
#endif
}

TEST_CASE("tinydpcppnn::encoding Grid Encoding") {
    SUBCASE("Test grid encoding using create_grid_encoding instead of factory") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 1;
        const int padded_output_width = 32;
        sycl::queue q;

        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(1.0f).wait();
        auto input_view = input.GetView();

        DeviceMatrix<float> output_float(batch_size, padded_output_width, q);
        output_float.fill(1.23f).wait(); // fill with something to check if it is written to
        auto output_view = output_float.GetView();

        const json encoding_config{
            {EncodingParams::N_DIMS_TO_ENCODE, input_width}, {EncodingParams::ENCODING, EncodingNames::GRID},
            {EncodingParams::GRID_TYPE, GridType::Hash},     {EncodingParams::N_LEVELS, 16},
            {EncodingParams::N_FEATURES_PER_LEVEL, 2},       {EncodingParams::LOG2_HASHMAP_SIZE, 19},
            {EncodingParams::BASE_RESOLUTION, 16},           {EncodingParams::PER_LEVEL_SCALE, 2.0}};

        std::shared_ptr<GridEncoding<float>> network =
            tinydpcppnn::encodings::grid::create_grid_encoding<float>(encoding_config, padded_output_width, q);
        q.wait();

        CHECK(network->get_n_params() > 0);
        std::vector<float> tmp_params_host(network->get_n_params(), 1.0f);
        initialize_arange(tmp_params_host);

        network->get_params()->copy_from_host(tmp_params_host);

        std::unique_ptr<Context> model_ctx = network->forward_impl(input_view, &output_view);
        q.wait();

        const std::vector<float> out = output_float.copy_to_host();
        const std::vector<float> reference_out = {
            -1,      -1,      -0.9985, -0.9985, -0.98,   -0.98,   -0.8076, -0.8076, -0.6606, -0.6606, -0.5107,
            -0.5107, -0.4202, -0.4202, -0.2527, -0.2527, -0.1031, -0.1031, 0.06964, 0.06964, 0.1893,  0.1893,
            0.2996,  0.2996,  0.4565,  0.4565,  0.6128,  0.6128,  0.7783,  0.7783,  0.9258,  0.9258};

        // Check if the actual vector is equal to the expected vector within the tolerance
        CHECK(areVectorsWithinTolerance(out, reference_out, 1.0e-3));
    }
    SUBCASE("Test grid encoding backward 0 values") {
        // SWIFTNET
        const int input_width = 3;
        const int batch_size = 1;
        const int padded_output_width = 32;
        sycl::queue q;

        DeviceMatrix<float> input(batch_size, input_width, q);
        input.fill(1.0f).wait();

        DeviceMatrix<float> output_float(batch_size, padded_output_width, q);
        output_float.fill(0.0f).wait(); // fill with something to check if it is written to

        const json encoding_config{
            {EncodingParams::N_DIMS_TO_ENCODE, input_width}, {EncodingParams::ENCODING, EncodingNames::GRID},
            {EncodingParams::GRID_TYPE, GridType::Hash},     {EncodingParams::N_LEVELS, 16},
            {EncodingParams::N_FEATURES_PER_LEVEL, 2},       {EncodingParams::LOG2_HASHMAP_SIZE, 19},
            {EncodingParams::BASE_RESOLUTION, 16},           {EncodingParams::PER_LEVEL_SCALE, 2.0}};

        std::shared_ptr<GridEncoding<float>> encoding =
            tinydpcppnn::encodings::grid::create_grid_encoding<float>(encoding_config, padded_output_width, q);
        q.wait();

        DeviceMatrix<float> gradients(encoding->get_n_params(), 1, q);
        gradients.fill(0.123f).wait(); // fill with something to check if it is written to
        auto gradients_view = gradients.GetView();

        encoding->get_params()->fill(1.0f).wait();

        std::unique_ptr<Context> model_ctx = nullptr;
        DeviceMatrix<float> dL_doutput(batch_size, padded_output_width, q);
        dL_doutput.fill(0.0f).wait();
        auto input_view = input.GetView();
        auto dL_doutput_view = dL_doutput.GetView();

        encoding->backward_impl(*model_ctx, input_view, dL_doutput_view, &gradients_view);
        q.wait();

        CHECK(isVectorWithinTolerance(gradients.copy_to_host(), 0.0f, 1e-3));
        CHECK(isVectorWithinTolerance(encoding->get_params()->copy_to_host(), 1.0f, 1e-3));
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

TEST_CASE("tinydpcppnn::encoding bad configs") {

    SUBCASE("Not existing keyword") {
        const int batch_size = 2;
        const int input_width = 3;
        const int output_width = 3;

        sycl::queue q;

        // Define the parameters for creating IdentityEncoding
        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                                   {EncodingParams::SCALE, 1.0},
                                   {EncodingParams::OFFSET, 0.0},
                                   {EncodingParams::ENCODING, EncodingNames::IDENTITY},
                                   {"Bad string", 1.0}};
        CHECK_NOTHROW(create_encoding<float>(encoding_config, q, output_width));
    }
}