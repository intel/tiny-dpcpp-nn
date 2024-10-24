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
#include <cmath>

#include "doctest/doctest.h"
#include "encoding_factory.h"
#include "io.h"
#include "result_check.h"

using bf16 = sycl::ext::oneapi::bfloat16;
using tinydpcppnn::encodings::grid::GridEncoding;
using json = nlohmann::json;

template <typename T>
void initialize_arange(std::vector<T> &vec) {
  // Repeat the col_vector and perform the operations
  double offset = (double)vec.size() / 2.0;

  for (long long i = 0; i < vec.size(); i++) {
    vec[i] = static_cast<T>(i / offset - 1.0);
  }
}

template <typename T>
void test_encoding_with_no_params_forward_from_loaded_file(
    const int batch_size, const int input_width, const int output_width,
    std::string filepath, sycl::queue &q) {
  DeviceMatrix<float> input(batch_size, input_width, q);
  input.fill(0.0f).wait();

  DeviceMatrix<T> output(batch_size, output_width, q);
  output.fill(0.0f).wait();
  auto output_view = output.GetView();

  json config = io::loadJsonConfig(filepath + "/encoding_config.json");
  config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;

  std::shared_ptr<Encoding<T>> encoding = create_encoding<T>(config);
  encoding->set_padded_output_width(output_width);

  std::vector<T> params =
      io::loadVectorFromCSV<T>(filepath + "encoding_params.csv");
  CHECK(params.size() == 0);
  std::vector<float> input_ref =
      io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
  std::vector<T> output_ref =
      io::loadVectorFromCSV<T>(filepath + "output_encoding.csv");

  CHECK(input_ref.size() == input.size());
  CHECK(output_ref.size() == output.size());

  input.copy_from_host(input_ref).wait();
  auto input_view = input.GetView();

  std::unique_ptr<Context> model_ctx =
      encoding->forward_impl(&q, input_view, &output_view);
  q.wait();

  // Check if the actual vector is equal to the expected vector within the
  // tolerance
  CHECK(areVectorsWithinTolerance(output.copy_to_host(), output_ref, 1.0e-2));
}

template <typename T>
void test_encoding_training_from_loaded_file(const int batch_size,
                                             const int input_width,
                                             const int output_width,
                                             std::string filepath,
                                             sycl::queue &q) {
  DeviceMatrix<float> input(batch_size, input_width, q);
  input.fill(0.0f).wait();

  DeviceMatrix<T> output(batch_size, output_width, q);
  output.fill(0.0f).wait();
  auto output_view = output.GetView();

  json config = io::loadJsonConfig(filepath + "/encoding_config.json");
  config[EncodingParams::N_DIMS_TO_ENCODE] = input_width;

  std::shared_ptr<Encoding<T>> encoding =
      create_encoding<T>(config, output_width, q);

  std::vector<T> params =
      io::loadVectorFromCSV<T>(filepath + "encoding_params.csv");
  std::vector<float> input_ref =
      io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
  std::vector<T> output_ref =
      io::loadVectorFromCSV<T>(filepath + "output_encoding.csv");

  if (!encoding->get_n_params()) {
    throw std::runtime_error(
        "n_params is 0, DeviceMem cannot be created with size 0 and for "
        "training, we require params to be trained");
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

  std::unique_ptr<Context> model_ctx =
      encoding->forward_impl(&q, input_view, &output_view);
  q.wait();

  // Check if the actual vector is equal to the expected vector within the
  // tolerance
  CHECK(areVectorsWithinTolerance(output.copy_to_host(), output_ref, 1.0e-2));

  // testing backward now
  std::vector<T> params_grad_ref =
      io::loadVectorFromCSV<T>(filepath + "params_grad.csv");
  std::vector<T> dL_doutput_ref =
      io::loadVectorFromCSV<T>(filepath + "dL_doutput.csv");

  DeviceMatrix<T> dL_doutput(batch_size, output_width, q);
  dL_doutput.copy_from_host(dL_doutput_ref).wait();
  auto dL_doutput_view = dL_doutput.GetView();

  encoding->backward_impl(&q, *model_ctx, input_view, dL_doutput_view,
                          &gradients_view);
  q.wait();

  std::vector<float> enc_params(encoding->get_n_params());
  std::vector<float> enc_grads = gradients.copy_to_host();

  q.memcpy(enc_params.data(), encoding->get_params(),
           encoding->get_n_params() * sizeof(float))
      .wait();
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
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, input_width},
        {EncodingParams::SCALE, 1.0},
        {EncodingParams::OFFSET, 0.0},
        {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
    std::shared_ptr<Encoding<float>> network =
        create_encoding<float>(encoding_config, q, output_width);
    std::unique_ptr<Context> model_ctx =
        network->forward_impl(input.GetView(), &output_view);
    q.wait();

    std::vector<float> in = input.copy_to_host();
    std::vector<float> out = output_float.copy_to_host();

    const float epsilon =
        1e-3;  // Set the tolerance for floating-point comparisons

    // Check if the actual vector is equal to the expected vector within the
    // tolerance
    for (size_t i = 0; i < out.size(); ++i) {
      CHECK(static_cast<float>(in[i]) ==
            doctest::Approx(out[i]).epsilon(epsilon));
    }
  }
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
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, input_width},
        {EncodingParams::DEGREE, DEGREE},
        {EncodingParams::ENCODING, EncodingNames::SPHERICALHARMONICS}};
    std::shared_ptr<Encoding<float>> network =
        create_encoding<float>(encoding_config, q, output_width);
    std::unique_ptr<Context> model_ctx =
        network->forward_impl(input.GetView(), &output_view);
    q.wait();

    std::vector<float> out = output_float.copy_to_host();
    const std::vector<float> reference_out = {0.2821, 1.0, 1.0,
                                              0.2821, 1.0, 1.0};

    const double epsilon = 1e-3;
    // Check if the actual vector is equal to the expected vector within the
    // tolerance
    CHECK(areVectorsWithinTolerance(out, reference_out, epsilon));
  }
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
    output_float.fill(1.23f)
        .wait();  // fill with something to check if it is written to
    auto output_view = output_float.GetView();

    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                               {EncodingParams::ENCODING, EncodingNames::GRID},
                               {EncodingParams::GRID_TYPE, GridType::Hash},
                               {EncodingParams::N_LEVELS, 16},
                               {EncodingParams::N_FEATURES_PER_LEVEL, 2},
                               {EncodingParams::LOG2_HASHMAP_SIZE, 19},
                               {EncodingParams::BASE_RESOLUTION, 16},
                               {EncodingParams::PER_LEVEL_SCALE, 2.0}};

    std::shared_ptr<GridEncoding<float>> network =
        tinydpcppnn::encodings::grid::create_grid_encoding<float>(
            encoding_config, padded_output_width, q);
    q.wait();

    CHECK(network->get_n_params() > 0);
    std::vector<float> tmp_params_host(network->get_n_params(), 1.0f);
    initialize_arange(tmp_params_host);

    network->get_params()->copy_from_host(tmp_params_host);

    std::unique_ptr<Context> model_ctx =
        network->forward_impl(input_view, &output_view);
    q.wait();

    const std::vector<float> out = output_float.copy_to_host();
    const std::vector<float> reference_out = {
        -1,      -1,      -0.9985, -0.9985, -0.98,   -0.98,   -0.8076, -0.8076,
        -0.6606, -0.6606, -0.5107, -0.5107, -0.4202, -0.4202, -0.2527, -0.2527,
        -0.1031, -0.1031, 0.06964, 0.06964, 0.1893,  0.1893,  0.2996,  0.2996,
        0.4565,  0.4565,  0.6128,  0.6128,  0.7783,  0.7783,  0.9258,  0.9258};

    // Check if the actual vector is equal to the expected vector within the
    // tolerance
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
    output_float.fill(0.0f)
        .wait();  // fill with something to check if it is written to

    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                               {EncodingParams::ENCODING, EncodingNames::GRID},
                               {EncodingParams::GRID_TYPE, GridType::Hash},
                               {EncodingParams::N_LEVELS, 16},
                               {EncodingParams::N_FEATURES_PER_LEVEL, 2},
                               {EncodingParams::LOG2_HASHMAP_SIZE, 19},
                               {EncodingParams::BASE_RESOLUTION, 16},
                               {EncodingParams::PER_LEVEL_SCALE, 2.0}};

    std::shared_ptr<GridEncoding<float>> encoding =
        tinydpcppnn::encodings::grid::create_grid_encoding<float>(
            encoding_config, padded_output_width, q);
    q.wait();

    std::cout << "encoding param dim: " << encoding->get_params()->rows()
              << ", " << encoding->get_params()->cols() << std::endl;

    DeviceMatrix<float> gradients(encoding->get_n_params(), 1, q);

    std::cout << "encoding gradients dim: " << gradients.rows() << ", "
              << gradients.cols() << std::endl;

    gradients.fill(0.123f)
        .wait();  // fill with something to check if it is written to
    auto gradients_view = gradients.GetView();

    encoding->get_params()->fill(1.0f).wait();

    std::unique_ptr<Context> model_ctx = nullptr;
    DeviceMatrix<float> dL_doutput(batch_size, padded_output_width, q);
    dL_doutput.fill(0.0f).wait();
    auto input_view = input.GetView();
    auto dL_doutput_view = dL_doutput.GetView();

    encoding->backward_impl(*model_ctx, input_view, dL_doutput_view,
                            &gradients_view);
    q.wait();

    CHECK(isVectorWithinTolerance(gradients.copy_to_host(), 0.0f, 1e-3));
    CHECK(isVectorWithinTolerance(encoding->get_params()->copy_to_host(), 1.0f,
                                  1e-3));
  }
}

///Correctness tests follow Mildenhall, 2020 equation 4 in Sec. 5.1
///where f(x) = (sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(n_fequ-1) * pi * x), cos(2^(n_fequ-1) * pi * x)), 
TEST_CASE("tinydpcppnn::encoding Frequency Encoding") {
  SUBCASE("create ok")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_NOTHROW(create_encoding<float>(encoding_config, q));
  }

  SUBCASE("create no n_frequencies")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_THROWS(create_encoding<float>(encoding_config, q));
  }

  SUBCASE("create n_frequencies 0")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 0},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_THROWS(create_encoding<float>(encoding_config, q));
  }

  SUBCASE("create n_frequencies < 0")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, -1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_THROWS(create_encoding<float>(encoding_config, q));
  }

  SUBCASE("create n_dims_to_encode 0")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 0},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_THROWS(create_encoding<float>(encoding_config, q));
  }

  SUBCASE("create too small padded")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_THROWS(create_encoding<float>(encoding_config, q, 1));
  }

  SUBCASE("create padded works")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_NOTHROW(create_encoding<float>(encoding_config, q, 2));
  }

  SUBCASE("create padded works 2")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    CHECK_NOTHROW(create_encoding<float>(encoding_config, q, 3));
  }

  SUBCASE("widths correct 1")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    auto encoding = create_encoding<float>(encoding_config, q);
    ///check if the encoding has the correct parameters
    CHECK(encoding->get_padded_output_width() == 2);
    CHECK(encoding->get_output_width() == 2);
    CHECK(encoding->get_n_params() == 0);
    CHECK(encoding->get_input_width() == 1);
  }

  SUBCASE("widths correct 2")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 2},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    auto encoding = create_encoding<float>(encoding_config, q);
    ///check if the encoding has the correct parameters
    CHECK(encoding->get_padded_output_width() == 4);
    CHECK(encoding->get_output_width() == 4);
    CHECK(encoding->get_n_params() == 0);
    CHECK(encoding->get_input_width() == 1);
  }

  SUBCASE("widths correct 3")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 3},
        {EncodingParams::N_FREQUENCIES, 2},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    auto encoding = create_encoding<float>(encoding_config, q);
    ///check if the encoding has the correct parameters
    CHECK(encoding->get_padded_output_width() == 12);
    CHECK(encoding->get_output_width() == 12);
    CHECK(encoding->get_n_params() == 0);
    CHECK(encoding->get_input_width() == 3);
  }

  SUBCASE("widths correct 4")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 3},
        {EncodingParams::N_FREQUENCIES, 2},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    auto encoding = create_encoding<float>(encoding_config, q, 24);
    ///check if the encoding has the correct parameters
    CHECK(encoding->get_padded_output_width() == 24);
    CHECK(encoding->get_output_width() == 12);
    CHECK(encoding->get_n_params() == 0);
    CHECK(encoding->get_input_width() == 3);
  }

  SUBCASE("forward not padded")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    auto encoding = create_encoding<float>(encoding_config, q);

    //setup input and output matrices
    constexpr size_t batch_size = 1;
    constexpr float input_val = 1.234f;
    DeviceMatrix<float> input(batch_size, 1, q);
    input.fill(input_val).wait();
    DeviceMatrix<float> output(batch_size, encoding->get_padded_output_width(), q);
    DeviceMatrixView<float> out_view = output.GetView();
    
    //run the forward pass
    CHECK_NOTHROW(encoding->forward_impl(input.GetView(), &out_view));
    
    //copy result to host 
    auto out_host = output.copy_to_host();

    //setup expected values
    std::vector<float> expected{std::sin((float)M_PI*input_val), std::cos((float)M_PI*input_val)};

    //Check if result is same as expected
    CHECK(areVectorsWithinTolerance(out_host, expected, 1e-4));
  }

  SUBCASE("forward padded")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 1},
        {EncodingParams::N_FREQUENCIES, 1},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    auto encoding = create_encoding<float>(encoding_config, q, 4);

    //setup input and output matrices
    constexpr size_t batch_size = 1;
    constexpr float input_val = 1.234f;
    DeviceMatrix<float> input(batch_size, 1, q);
    input.fill(input_val).wait();
    DeviceMatrix<float> output(batch_size, encoding->get_padded_output_width(), q);
    DeviceMatrixView<float> out_view = output.GetView();
    
    //run the forward pass
    CHECK_NOTHROW(encoding->forward_impl(input.GetView(), &out_view));
    
    //copy result to host 
    auto out_host = output.copy_to_host();

    //setup expected values
    std::vector<float> expected{std::sin((float)M_PI*input_val), std::cos((float)M_PI*input_val), 1, 1};

    //Check if result is same as expected
    CHECK(areVectorsWithinTolerance(out_host, expected, 1e-4));
  }

    SUBCASE("forward larger padded")
  {
    sycl::queue q;
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, 3},
        {EncodingParams::N_FREQUENCIES, 2},
        {EncodingParams::ENCODING, EncodingNames::FREQUENCY}};
    auto encoding = create_encoding<float>(encoding_config, q, 14);

    //setup input and output matrices
    constexpr size_t batch_size = 2;
    std::vector<float> input_vals{0,1,2,3,4,5};
    DeviceMatrix<float> input(batch_size, 3, q);
    input.copy_from_host(input_vals).wait();
    DeviceMatrix<float> output(batch_size, encoding->get_padded_output_width(), q);
    DeviceMatrixView<float> out_view = output.GetView();
    
    //run the forward pass
    CHECK_NOTHROW(encoding->forward_impl(input.GetView(), &out_view));
    
    //copy result to host 
    auto out_host = output.copy_to_host();

    //setup expected values
    std::vector<float> expected{
        std::sin((float)M_PI*input_vals[0]), std::cos((float)M_PI*input_vals[0]), std::sin(2.0f*(float)M_PI*input_vals[0]), std::cos(2.0f*(float)M_PI*input_vals[0]), 
        std::sin((float)M_PI*input_vals[1]), std::cos((float)M_PI*input_vals[1]), std::sin(2.0f*(float)M_PI*input_vals[1]), std::cos(2.0f*(float)M_PI*input_vals[1]),
        std::sin((float)M_PI*input_vals[2]), std::cos((float)M_PI*input_vals[2]), std::sin(2.0f*(float)M_PI*input_vals[2]), std::cos(2.0f*(float)M_PI*input_vals[2]),
        1, 1,
        std::sin((float)M_PI*input_vals[3]), std::cos((float)M_PI*input_vals[3]), std::sin(2.0f*(float)M_PI*input_vals[3]), std::cos(2.0f*(float)M_PI*input_vals[3]), 
        std::sin((float)M_PI*input_vals[4]), std::cos((float)M_PI*input_vals[4]), std::sin(2.0f*(float)M_PI*input_vals[4]), std::cos(2.0f*(float)M_PI*input_vals[4]),
        std::sin((float)M_PI*input_vals[5]), std::cos((float)M_PI*input_vals[5]), std::sin(2.0f*(float)M_PI*input_vals[5]), std::cos(2.0f*(float)M_PI*input_vals[5]),
        1, 1};

    //Check if result is same as expected
    CHECK(areVectorsWithinTolerance(out_host, expected, 1e-4));
  }

}
          

TEST_CASE("tinydpcppnn::encoding bad configs") {
  SUBCASE("Not existing keyword") {
    const int batch_size = 2;
    const int input_width = 3;
    const int output_width = 3;

    sycl::queue q;

    // Define the parameters for creating IdentityEncoding
    const json encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, input_width},
        {EncodingParams::SCALE, 1.0},
        {EncodingParams::OFFSET, 0.0},
        {EncodingParams::ENCODING, EncodingNames::IDENTITY},
        {"Bad string", 1.0}};
    CHECK_NOTHROW(create_encoding<float>(encoding_config, q, output_width));
  }
}