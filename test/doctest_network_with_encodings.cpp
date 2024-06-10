/**
 * @file doctest_network_with_encodings.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the network with encodings class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "doctest/doctest.h"
#include "io.h"
#include "l2.h"
#include "mlp.h"
#include "network_with_encodings.h"
#include "result_check.h"

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;
using tinydpcppnn::encodings::grid::GridEncoding;
using json = nlohmann::json;

template <typename T>
std::vector<T> create_target_ref_vector(int output_width, T target_val, int unpadded_output_width) {
    if (output_width > unpadded_output_width) {
        throw std::invalid_argument("output_width cannot be greater than N");
    }

    // Initialize a vector of N zeros
    std::vector<T> target_ref(unpadded_output_width, 0);
    // Set the first output_width elements to target_val
    std::fill_n(target_ref.begin(), output_width, target_val);
    return target_ref;
}

/// Function which applies a grid encoding to a R2 vector, resulting in a vector of size
/// network_input_width, then applies the network and the output is the network_output_width
// ATTENTION: currently only works for WIDTH=64
template <typename T_enc, typename T_net, int WIDTH = 64>
void test_network_with_encoding_inference_loaded(sycl::queue &q, std::string filepath, const int n_hidden_layers,
                                                 const int batch_size, const int unpadded_output_width,
                                                 const int encoding_input_width) {

    json encoding_config = io::loadJsonConfig(filepath + "encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = encoding_input_width;

    auto Net = create_network_with_encoding<T_enc, T_net, WIDTH>(q, unpadded_output_width, n_hidden_layers,
                                                                 Activation::ReLU, Activation::None, encoding_config);

    std::vector<T_enc> encoding_params = io::loadVectorFromCSV<T_enc>(filepath + "encoding_params.csv");
    std::vector<T_net> network_params = io::load_weights_as_packed_from_file<T_net, WIDTH>(
        filepath + "network_params.csv", n_hidden_layers, WIDTH, WIDTH);
    Net->set_network_params(network_params); // not using initialize_params, as those aren't set when identity is set
    std::unique_ptr<DeviceMatrix<T_enc>> params_full_precision_ptr;
    if (encoding_params.size()) {
        params_full_precision_ptr = std::make_unique<DeviceMatrix<T_enc>>(Net->get_encoding()->n_params(), 1, q);
        params_full_precision_ptr->fill(1.0f).wait();

        Net->initialize_params(*params_full_precision_ptr, &encoding_params);
    }

    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    std::vector<float> input_encoding_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    input_encoding.copy_from_host(input_encoding_ref);
    q.wait();

    DeviceMatrix<T_enc> output_encoding(batch_size, Net->get_network()->get_input_width(), q);
    DeviceMatrix<T_net> input_network(batch_size, Net->get_network()->get_input_width(), q);

    DeviceMatrix<T_net> output_network(batch_size, Net->get_network()->get_output_width(), q);
    auto output_network_views = output_network.GetViews();
    Net->inference(input_encoding.GetView(), input_network, output_encoding, output_network_views, {});
    q.wait();
    std::vector<T_enc> output_encoding_vec = output_encoding.copy_to_host();
    std::vector<T_net> output_network_vec = output_network.copy_to_host();

    std::vector<T_enc> encoding_output_ref = io::loadVectorFromCSV<T_enc>(filepath + "output_encoding.csv");
    if (!areVectorsWithinTolerance(output_encoding_vec, encoding_output_ref, 1.0e-2)) {
        printVector("output_encoding_vec", output_encoding_vec);
        printVector("encoding_output_ref", encoding_output_ref);
    }
    CHECK(areVectorsWithinTolerance(output_encoding_vec, encoding_output_ref, 1.0e-2));

    std::vector<T_net> network_output_ref = io::loadVectorFromCSV<T_net>(filepath + "output_network.csv");
    if (!areVectorsWithinTolerance(output_network_vec, network_output_ref, 1.0e-2)) {
        printVector("output_network_vec", output_network_vec);
        printVector("network_output_ref", network_output_ref);
    }
    CHECK(areVectorsWithinTolerance(output_network_vec, network_output_ref, 1.0e-2));
}

/// Function which applies a grid encoding to a R2 vector, resulting in a vector of size
/// network_input_width, then applies the network and the output is the network_output_width
// ATTENTION: currently only works for WIDTH=64
template <typename T_enc, typename T_net, int WIDTH = 64>
void test_network_with_encoding_forward_loaded(sycl::queue &q, std::string filepath, const int n_hidden_layers,
                                               const int batch_size, const int unpadded_output_width,
                                               const int encoding_input_width) {

    json encoding_config = io::loadJsonConfig(filepath + "encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = encoding_input_width;

    auto Net = create_network_with_encoding<T_enc, T_net, WIDTH>(q, unpadded_output_width, n_hidden_layers,
                                                                 Activation::ReLU, Activation::None, encoding_config);

    std::vector<T_enc> encoding_params = io::loadVectorFromCSV<T_enc>(filepath + "encoding_params.csv");
    std::vector<T_net> network_params = io::load_weights_as_packed_from_file<T_net, WIDTH>(
        filepath + "network_params.csv", n_hidden_layers, Net->get_network()->get_input_width(),
        Net->get_network()->get_output_width());
    Net->set_network_params(network_params); // not using initialize_params, as those aren't set when identity is set
    std::unique_ptr<DeviceMem<T_enc>> gradients_ptr;
    std::unique_ptr<DeviceMem<T_enc>> params_full_precision_ptr;
    if (encoding_params.size()) {

        gradients_ptr = std::make_unique<DeviceMem<T_enc>>(Net->get_encoding()->n_params(), q);
        params_full_precision_ptr = std::make_unique<DeviceMem<T_enc>>(Net->get_encoding()->n_params(), q);

        gradients_ptr->fill(1.0f).wait();
        params_full_precision_ptr->fill(1.0f).wait();

        Net->initialize_params(*gradients_ptr, *params_full_precision_ptr, &encoding_params);
    }

    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    std::vector<float> input_encoding_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    input_encoding.copy_from_host(input_encoding_ref);
    q.wait();

    DeviceMatrix<T_enc> output_encoding(batch_size, Net->get_network()->get_input_width(), q);
    DeviceMatrix<T_net> input_network(batch_size, Net->get_network()->get_input_width(), q);

    DeviceMatrices<T_net> interm_forw(
        Net->get_network()->get_n_hidden_layers() + 2, batch_size, Net->get_network()->get_network_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    Net->forward_pass(input_encoding, input_network, output_encoding, interm_forw, {});
    q.wait();

    auto interm_forw_vec = interm_forw.copy_to_host();

    std::vector<T_enc> output_encoding_vec = output_encoding.copy_to_host();
    std::vector<T_net> output_network_vec(interm_forw_vec.end() - batch_size * Net->get_network()->get_output_width(),
                                          interm_forw_vec.end());

    std::vector<T_enc> encoding_output_ref = io::loadVectorFromCSV<T_enc>(filepath + "output_encoding.csv");
    std::vector<T_net> network_output_ref = io::loadVectorFromCSV<T_net>(filepath + "output_network.csv");

    CHECK(areVectorsWithinTolerance(output_encoding_vec, encoding_output_ref, 1.0e-2));
    CHECK(areVectorsWithinTolerance(output_network_vec, network_output_ref, 1.0e-2));
}

template <typename T_enc, typename T_net, int WIDTH = 64>
void test_network_with_encoding_training_loaded(sycl::queue &q, std::string filepath, const int n_hidden_layers,
                                                const int batch_size, const int unpadded_output_width,
                                                const int encoding_input_width) {
    float epsilon = 1e-2;
    json encoding_config = io::loadJsonConfig(filepath + "encoding_config.json");
    encoding_config[EncodingParams::N_DIMS_TO_ENCODE] = encoding_input_width;

    auto Net = create_network_with_encoding<T_enc, T_net, WIDTH>(q, unpadded_output_width, n_hidden_layers,
                                                                 Activation::ReLU, Activation::None, encoding_config);

    std::vector<T_enc> encoding_params_ref = io::loadVectorFromCSV<T_enc>(filepath + "encoding_params.csv");
    std::vector<T_net> network_params = io::load_weights_as_packed_from_file<T_net, WIDTH>(
        filepath + "network_params.csv", n_hidden_layers, Net->get_network()->get_input_width(),
        Net->get_network()->get_output_width());
    Net->set_network_params(network_params);
    std::unique_ptr<DeviceMatrix<T_enc>> enc_gradients_ptr;
    std::unique_ptr<DeviceMatrix<T_enc>> params_full_precision_ptr;
    std::unique_ptr<DeviceMem<T_enc>> all_gradients_ptr;

    DeviceMatrices<T_net> network_gradient(Net->get_network()->get_n_hidden_layers() + 1,
                                           Net->get_network()->get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                                           Net->get_network()->get_output_width(), q);
    if (encoding_params_ref.size()) {
        enc_gradients_ptr = std::make_unique<DeviceMatrix<T_enc>>(Net->get_encoding()->n_params(), 1, q);
        enc_gradients_ptr->fill(1.234f).wait();

        params_full_precision_ptr = std::make_unique<DeviceMatrix<T_enc>>(Net->get_encoding()->n_params(), 1, q);
        params_full_precision_ptr->fill(3.456f).wait();

        all_gradients_ptr =
            std::make_unique<DeviceMem<T_enc>>(network_gradient.nelements() + enc_gradients_ptr->n_elements(), q);
        all_gradients_ptr->fill(2.345f).wait();
        Net->initialize_params(*params_full_precision_ptr, &encoding_params_ref);
    }

    DeviceMatrix<float> input_encoding(batch_size, encoding_input_width, q);
    std::vector<float> input_encoding_ref = io::loadVectorFromCSV<float>(filepath + "input_encoding.csv");
    input_encoding.copy_from_host(input_encoding_ref);
    q.wait();

    DeviceMatrix<T_enc> output_encoding(batch_size, Net->get_network()->get_input_width(), q);
    DeviceMatrix<T_net> input_network(batch_size, Net->get_network()->get_input_width(), q);

    DeviceMatrices<T_net> interm_forw(
        Net->get_network()->get_n_hidden_layers() + 2, batch_size, Net->get_network()->get_network_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    Net->forward_pass(input_encoding.GetView(), input_network, output_encoding, interm_forw.GetViews(), {});
    q.wait();
    auto interm_forw_vec = interm_forw.copy_to_host();

    std::vector<T_enc> output_encoding_vec = output_encoding.copy_to_host();
    std::vector<T_net> output_network_vec(interm_forw_vec.end() - batch_size * Net->get_network()->get_output_width(),
                                          interm_forw_vec.end());

    std::vector<T_enc> encoding_output_ref = io::loadVectorFromCSV<T_enc>(filepath + "output_encoding.csv");
    std::vector<T_net> network_output_ref = io::loadVectorFromCSV<T_net>(filepath + "output_network.csv");
    CHECK(areVectorsWithinTolerance(output_encoding_vec, encoding_output_ref, epsilon));
    CHECK(areVectorsWithinTolerance(output_network_vec, network_output_ref, epsilon));
    // load loss here and calculate
    L2Loss<T_net> l2_loss;
    // using network_output_ref as network_output as they passed tolerance test (< 1%)
    DeviceMatrix<T_net> network_output(batch_size, Net->get_network()->get_output_width(), q);
    network_output.fill(0.0f).wait();
    network_output.copy_from_host(network_output_ref).wait();

    T_net loss_scale = 1.0;

    DeviceMatrix<T_net> dL_doutput(batch_size, Net->get_network()->get_output_width(), q);
    dL_doutput.fill(0.0f).wait();

    DeviceMatrix<float> targets(batch_size, Net->get_network()->get_output_width(), q);
    targets.fill(0.0f).wait();
    std::vector<float> targets_ref = io::loadVectorFromCSV<float>(filepath + "targets.csv");
    targets.copy_from_host(targets_ref).wait();

    DeviceMatrix<float> loss(batch_size, Net->get_network()->get_output_width(), q);
    loss.fill(0.0f).wait();

    sycl::event sycl_event = l2_loss.evaluate(q, loss_scale, network_output.GetView(), targets, loss, dL_doutput);
    q.wait();

    std::vector<T_net> dL_doutput_ref = io::loadVectorFromCSV<T_net>(filepath + "dL_doutput.csv");
    std::vector<T_net> dL_doutput_vec = dL_doutput.copy_to_host();

    if (!areVectorsWithinTolerance(dL_doutput_ref, dL_doutput_vec, epsilon)) {
        printVector("dL_doutput_ref", dL_doutput_ref);
        printVector("dL_doutput_vec", dL_doutput_vec);
    }
    CHECK(areVectorsWithinTolerance(dL_doutput_ref, dL_doutput_vec, epsilon));

    DeviceMatrices<T_net> interm_backw(
        Net->get_network()->get_n_hidden_layers() + 1, batch_size, Net->get_network()->get_input_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    // saving intermediate dL_dinput of swiftnet here, that is input to grid
    DeviceMatrix<T_net> dL_dinput(batch_size, Net->get_network()->get_input_width(), q);
    network_gradient.fill(0.0f).wait();
    interm_backw.fill(0.0f).wait();
    dL_dinput.fill(1.234f).wait();
    auto dL_dinput_view = dL_dinput.GetView();
    // CHECK params before backward pass
    std::vector<T_enc> enc_params(Net->get_encoding()->n_params());
    if (Net->get_encoding()->n_params()) {
        Net->backward_pass(dL_doutput.GetView(), network_gradient.GetViews(), enc_gradients_ptr->GetView(),
                           interm_backw.GetViews(), interm_forw.GetViews(), {}, input_encoding.GetView(),
                           &dL_dinput_view);
    } else {
        Net->backward_pass(dL_doutput.GetView(), network_gradient.GetViews(), interm_backw.GetViews(),
                           interm_forw.GetViews(), {}, &dL_dinput_view);
    }

    std::vector<T_net> dL_dinput_ref = io::loadVectorFromCSV<T_net>(filepath + "dL_dinput.csv");
    std::vector<T_net> dL_dinput_vec = dL_dinput.copy_to_host();

    if (!areVectorsWithinTolerance(dL_dinput_vec, dL_dinput_ref, epsilon)) {
        double sum_vec = 0.0;
        double sum_ref = 0.0;
        for (auto &val : dL_dinput_ref) {
            sum_ref += val;
        }
        for (auto &val : dL_dinput_vec) {
            sum_vec += val;
        }
        std::cout << "dL_dinput_ref sum: " << sum_ref << std::endl;
        std::cout << "dL_dinput_vec sum: " << sum_vec << std::endl;
        printVector("dL_dinput_ref", dL_dinput_ref);
        printVector("dL_dinput_vec", dL_dinput_vec);
    }
    CHECK(areVectorsWithinTolerance(dL_dinput_vec, dL_dinput_ref, epsilon));

    // // network param grad
    // std::vector<T_net> network_params_grad_vec = network_gradient.copy_to_host();
    // std::vector<T_net> network_params_grad_ref = io::loadVectorFromCSV<T_net>(filepath + "network_params_grad.csv");
    // if (!areVectorsWithinTolerance(network_params_grad_vec, network_params_grad_ref, epsilon)) {
    //     double sum_vec = 0.0;
    //     double sum_ref = 0.0;
    //     for (auto &val : network_params_grad_ref) {
    //         sum_ref += val;
    //     }
    //     for (auto &val : network_params_grad_vec) {
    //         sum_vec += val;
    //     }
    //     std::cout << "network_params_grad_ref sum: " << sum_ref << std::endl;
    //     std::cout << "network_params_grad_vec sum: " << sum_vec << std::endl;
    // }
    // CHECK(areVectorsWithinTolerance(network_params_grad_vec, network_params_grad_ref, epsilon));

    // if (encoding_config[EncodingParams::ENCODING] == EncodingNames::GRID) {
    //     // encoding param grad
    //     q.memcpy(enc_params.data(), Net->get_encoding()->params(), Net->get_encoding()->n_params() * sizeof(T_enc))
    //         .wait();
    //     CHECK(areVectorsWithinTolerance(enc_params, encoding_params_ref, epsilon));

    //     // CHECK params after backward pass it shouldn't have changed because no optimise step
    //     std::vector<T_enc> encoding_params_grad_ref =
    //         io::loadVectorFromCSV<T_enc>(filepath + "encoding_params_grad.csv");
    //     std::vector<float> encoding_params_grad_vec = enc_gradients_ptr->copy_to_host();
    //     if (!areVectorsWithinTolerance(encoding_params_grad_vec, encoding_params_grad_ref, epsilon)) {
    //         double sum_vec = 0.0;
    //         double sum_ref = 0.0;
    //         for (auto &val : encoding_params_grad_ref) {
    //             sum_ref += val;
    //         }
    //         for (auto &val : encoding_params_grad_vec) {
    //             sum_vec += val;
    //         }
    //         std::cout << "encoding_params_grad_ref sum: " << sum_ref << std::endl;
    //         std::cout << "encoding_params_grad_vec sum: " << sum_vec << std::endl;
    //     }
    //     CHECK(areVectorsWithinTolerance(encoding_params_grad_vec, encoding_params_grad_ref, epsilon));
    // }
}

template <int WIDTH>
void test_network_with_encoding_identity_inference(sycl::queue &q, const int input_width, const int output_width) {
    const int batch_size = 8;
    const int n_hidden_layers = 1;
    // Define the parameters for creating IdentityEncoding
    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                               {EncodingParams::SCALE, 1.0},
                               {EncodingParams::OFFSET, 0.0},
                               {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
    auto Net = create_network_with_encoding<float, bf16, WIDTH>(q, output_width, n_hidden_layers, Activation::ReLU,
                                                                Activation::None, encoding_config);

    const bf16 weight_val = 0.01;
    std::vector<bf16> new_weights(Net->get_network()->get_weights_matrices().nelements(), weight_val);

    Net->set_network_params(new_weights);

    constexpr float input_val = 1.0f;
    DeviceMatrix<float> input_encoding(batch_size, input_width, q);
    input_encoding.fill(input_val).wait();

    DeviceMatrix<float> output_encoding = Net->GenerateEncodingOutputMatrix(batch_size);
    output_encoding.fill(0.0f).wait();
    DeviceMatrix<bf16> input_network(batch_size, Net->get_network()->get_input_width(), q);

    DeviceMatrix<bf16> output_network = Net->GenerateForwardOutputMatrix(batch_size);
    output_network.fill(1.234f).wait();
    auto output_network_views = output_network.GetViews();
    Net->inference(input_encoding.GetView(), input_network, output_encoding, output_network_views, {});
    q.wait();

    CHECK(isVectorWithinTolerance(output_network.copy_to_host(),
                                  input_val * std::pow(WIDTH * (double)weight_val, n_hidden_layers + 1), 1.0e-3));
}

template <typename T, int WIDTH>
void test_network_with_encoding_identity_forward(sycl::queue &q, const int input_width, const int output_width,
                                                 const int n_hidden_layers, const int batch_size,
                                                 std::string activation, std::string weight_init_mode) {
    // main functionalities of backward and forward are tested in doctest_swifnet
    // here, we test only if the combination of encoding (tested in doctest_encoding) and swifnet works

    const float input_val = 1.0f;
    const float target_val = 0.1;
    std::vector<float> input_ref(input_width, input_val);

    CHECK(input_width == output_width);
    // this is not a hard requirement, but currently the loop over the
    // mlp reference (batch size = 1) assumes this. if this is changed, ensure the
    // checks are still correct
    CHECK(input_width == WIDTH);
    Activation network_activation;
    if (activation == "relu") {
        network_activation = Activation::ReLU;
    } else if (activation == "linear") {
        network_activation = Activation::None;
    }
    // Define the parameters for creating IdentityEncoding
    const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, input_width},
                               {EncodingParams::SCALE, 1.0},
                               {EncodingParams::OFFSET, 0.0},
                               {EncodingParams::ENCODING, EncodingNames::IDENTITY}};

    auto Net = create_network_with_encoding<float, bf16, WIDTH>(q, output_width, n_hidden_layers, network_activation,
                                                                Activation::None, encoding_config);

    mlp_cpp::MLP<float> mlp(Net->get_network()->get_input_width(), WIDTH, Net->get_network()->get_output_width(),
                            n_hidden_layers + 1, batch_size, activation, "linear", weight_init_mode);
    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<float, T>(mlp.getUnpackedWeights());

    Net->get_network()->set_weights_matrices(unpacked_weights, false);

    DeviceMatrix<float> input_encoding(batch_size, input_width, q);

    // Repeat source vector N times
    std::vector<float> input_full = mlp_cpp::stack_vector(input_ref, batch_size);
    input_encoding.copy_from_host(input_full).wait();

    DeviceMatrix<float> output_encoding = Net->GenerateEncodingOutputMatrix(batch_size);
    output_encoding.fill(0.0f).wait();
    DeviceMatrix<T> input_network(batch_size, Net->get_network()->get_input_width(), q);

    DeviceMatrices<T> interm_forw(
        Net->get_network()->get_n_hidden_layers() + 2, batch_size, Net->get_network()->get_input_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> interm_backw(
        Net->get_network()->get_n_hidden_layers() + 1, batch_size, Net->get_network()->get_network_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> network_backward_output(Net->get_network()->get_n_hidden_layers() + 1,
                                              Net->get_network()->get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                                              Net->get_network()->get_output_width(), q);

    Net->forward_pass(input_encoding.GetView(), input_network, output_encoding, interm_forw.GetViews(), {});
    q.wait();

    std::vector<std::vector<float>> fwd_result_ref = mlp.forward(input_ref, true);
    auto interm_forw_ref = mlp_cpp::repeat_inner_vectors<float>(fwd_result_ref, batch_size);
    std::vector<T> interm_forw_vec = interm_forw.copy_to_host();

    CHECK(interm_forw_vec.size() == interm_forw_ref.size());
    CHECK(areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2));
}

template <typename T, int WIDTH>
void test_network_with_encoding_backward(sycl::queue &q, const int input_width, const int output_width,
                                         const int n_hidden_layers, const int batch_size, std::string activation,
                                         std::string weight_init_mode, const json &encoding_config) {
    // main functionalities of backward and forward are tested in doctest_swifnet
    // here, we test only if the combination of encoding (tested in doctest_encoding) and swifnet works

    const float input_val = 1.0f;
    const float target_val = 0.1;
    std::vector<float> input_ref(input_width, input_val);
    const int unpadded_output_width = WIDTH;

    std::vector<float> target_ref = create_target_ref_vector<float>(output_width, target_val, unpadded_output_width);
    CHECK(input_width == WIDTH);
    Activation network_activation;
    if (activation == "relu") {
        network_activation = Activation::ReLU;
    } else if (activation == "linear") {
        network_activation = Activation::None;
    }

    mlp_cpp::MLP<float> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation, "linear",
                            weight_init_mode);
    auto Net = create_network_with_encoding<float, T, WIDTH>(q, output_width, n_hidden_layers, network_activation,
                                                             Activation::None, encoding_config);

    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<float, T>(mlp.getUnpackedWeights());
    Net->get_network()->set_weights_matrices(unpacked_weights, false);

    std::vector<mlp_cpp::Matrix<float>> grad_matrices_ref(n_hidden_layers + 1, mlp_cpp::Matrix<float>(1, 1));
    std::vector<std::vector<float>> loss_grads_ref;
    std::vector<float> loss_ref;

    std::vector<float> dL_doutput_ref;
    std::vector<float> dL_dinput_ref;

    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 1.0);

    DeviceMatrices<T> interm_forw(
        Net->get_network()->get_n_hidden_layers() + 2, batch_size, Net->get_network()->get_input_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> interm_backw(
        Net->get_network()->get_n_hidden_layers() + 1, batch_size, Net->get_network()->get_network_width(), batch_size,
        Net->get_network()->get_network_width(), batch_size, Net->get_network()->get_output_width(), q);

    DeviceMatrices<T> network_backward_output(Net->get_network()->get_n_hidden_layers() + 1,
                                              Net->get_network()->get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                                              Net->get_network()->get_output_width(), q);
    DeviceMatrix<T> dL_doutput(batch_size, Net->get_network()->get_output_width(), q);
    network_backward_output.fill(0.0).wait();
    dL_doutput.fill(0.0).wait();

    std::vector<std::vector<float>> fwd_result_ref = mlp.forward(input_ref, true);
    std::vector<float> interm_forw_ref = mlp_cpp::repeat_inner_vectors<float>(fwd_result_ref, batch_size);
    interm_forw.copy_from_host(mlp_cpp::convert_vector<float, T>(interm_forw_ref)).wait();

    std::vector<float> stacked_loss_grads_back_ref = mlp_cpp::stack_vector(loss_grads_ref.back(), batch_size);
    dL_doutput.copy_from_host(mlp_cpp::convert_vector<float, T>(stacked_loss_grads_back_ref)).wait();

    Net->get_network()->backward_pass(dL_doutput, network_backward_output, interm_backw, interm_forw, {});
    q.wait();
    std::vector<T> interm_backw_vec = interm_backw.copy_to_host();
    q.wait();
    for (int i = 0; i < loss_grads_ref.size(); i++) {
        std::vector<float> interm_backw_ref;
        std::vector<T> interm_backw_sliced_actual(interm_backw_vec.begin() + i * batch_size * WIDTH,
                                                  interm_backw_vec.begin() + i * batch_size * WIDTH +
                                                      batch_size * WIDTH);

        auto inner_stacked = mlp_cpp::stack_vector(loss_grads_ref[i], batch_size);
        for (T value : inner_stacked) {
            interm_backw_ref.push_back(value); // Add each element to the flattened vector
        }

        if (!areVectorsWithinTolerance(interm_backw_sliced_actual, interm_backw_ref, 1.0e-2)) {
            printVector("interm_backw_ref: ", interm_backw_ref);
            printVector("interm_backw_vec: ", interm_backw_sliced_actual);
        }
        CHECK(areVectorsWithinTolerance(interm_backw_sliced_actual, interm_backw_ref, 1.0e-2));
    }

    // flatten reference grad matrices
    std::vector<double> grads_ref;
    for (const auto &matrix : grad_matrices_ref) {
        for (size_t row = 0; row < matrix.rows(); ++row) {
            for (size_t col = 0; col < matrix.cols(); ++col) {
                grads_ref.push_back(matrix.data[row][col]);
            }
        }
    }

    auto grad_vec = network_backward_output.copy_to_host();
    bool grads_within_tolerance = areVectorsWithinTolerance(grad_vec, grads_ref, 1.0e-2);

    if (!grads_within_tolerance) {
        printVector("grads_ref", grads_ref);
        printVector("grad_vec", grad_vec);
    }
    CHECK(areVectorsWithinTolerance(grad_vec, grads_ref, 1.0e-2));
}
// Create a shared pointer of network with encoding using create_network_with_encoding
template <typename T_enc, typename T_net, int WIDTH = 64>
std::shared_ptr<NetworkWithEncoding<T_enc, T_net>> test_create_nwe_as_shared_ptr(sycl::queue &q,
                                                                                 const json &encoding_config) {

    static_assert(WIDTH == 64);
    constexpr int n_hidden_layers = 1;
    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr int unpadded_output_width = 1;
    Activation activation = Activation::ReLU;
    Activation output_activation = Activation::None;

    std::shared_ptr<NetworkWithEncoding<T_enc, T_net>> network_with_encoding_shared_ptr =
        create_network_with_encoding<T_enc, T_net, WIDTH>(q, unpadded_output_width, n_hidden_layers, activation,
                                                          output_activation, encoding_config);
    q.wait();

    assert(input_width == network_with_encoding_shared_ptr->get_network()->get_input_width());
    assert(output_width == network_with_encoding_shared_ptr->get_network()->get_output_width());
    return network_with_encoding_shared_ptr;
}

TEST_CASE("Network with Identity Encoding - test fwd") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;
    auto test_function = [=](auto T_type, sycl::queue &q, const int width, const int batch_size, std::string activation,
                             std::string weight_init_mode) {
        using T = decltype(T_type);
        if (width == 16)
            test_network_with_encoding_identity_forward<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation,
                                                               weight_init_mode);
        else if (width == 32)
            test_network_with_encoding_identity_forward<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation,
                                                               weight_init_mode);
        else if (width == 64)
            test_network_with_encoding_identity_forward<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation,
                                                               weight_init_mode);
        else if (width == 128)
            test_network_with_encoding_identity_forward<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation,
                                                                weight_init_mode);
        else
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

TEST_CASE("Network with Identity Encoding - test bwd") {
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
            test_network_with_encoding_backward<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, encoding_config);
        } else if (width == 32) {
            // Define the parameters for creating IdentityEncoding
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, encoding_config);
        } else if (width == 64) {
            // Define the parameters for creating IdentityEncoding
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, encoding_config);
        } else if (width == 128) {
            // Define the parameters for creating IdentityEncoding
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation,
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

TEST_CASE("Network with Grid Encoding - test network bwd only") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;
    const int input_width = 2;
    const json grid_encoding_config{
        {EncodingParams::N_DIMS_TO_ENCODE, input_width}, {EncodingParams::ENCODING, EncodingNames::GRID},
        {EncodingParams::GRID_TYPE, GridType::Hash},     {EncodingParams::N_LEVELS, 16},
        {EncodingParams::N_FEATURES_PER_LEVEL, 2},       {EncodingParams::LOG2_HASHMAP_SIZE, 19},
        {EncodingParams::BASE_RESOLUTION, 16},           {EncodingParams::PER_LEVEL_SCALE, 2.0}};

    auto test_function = [=](auto T_type, sycl::queue &q, const int width, const int batch_size, std::string activation,
                             std::string weight_init_mode) {
        using T = decltype(T_type);

        if (width == 32)
            test_network_with_encoding_backward<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, grid_encoding_config);
        else if (width == 64)
            test_network_with_encoding_backward<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, grid_encoding_config);
        else if (width == 128)
            test_network_with_encoding_backward<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation,
                                                        weight_init_mode, grid_encoding_config);
        else
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
                        if (width == 16) { // grid encoding outputs dim 32, which cannot be the input dim for 16 widths
                            SUBCASE(testName.c_str()) {
                                CHECK_THROWS(test_network_with_encoding_backward<sycl::ext::oneapi::bfloat16, 16>(
                                    q, 16, 16, n_hidden_layers, batch_size, activation, weight_init_mode,
                                    grid_encoding_config));
                            }
                        } else {
                            SUBCASE(testName.c_str()) {
                                CHECK_NOTHROW(test_function(type, q, width, batch_size, activation, weight_init_mode));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Network with Identity Encoding - test network bwd only padded") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;
    const int output_width = 5;

    auto test_function = [=](auto T_type, sycl::queue &q, const int width, const int batch_size, std::string activation,
                             std::string weight_init_mode) {
        using T = decltype(T_type);

        if (width == 16) {
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<T, 16>(q, 16, output_width, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, encoding_config);
        } else if (width == 32) {
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<T, 32>(q, 32, output_width, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, encoding_config);
        } else if (width == 64) {
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<T, 64>(q, 64, output_width, n_hidden_layers, batch_size, activation,
                                                       weight_init_mode, encoding_config);
        } else if (width == 128) {
            const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, width},
                                       {EncodingParams::SCALE, 1.0},
                                       {EncodingParams::OFFSET, 0.0},
                                       {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
            test_network_with_encoding_backward<T, 128>(q, 128, output_width, n_hidden_layers, batch_size, activation,
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
TEST_CASE("tinydpcppnn::network_with_encoding step-by-step") {
    sycl::queue q(gpu_selector_v);
    std::shared_ptr<NetworkWithEncoding<float, bf16>> net;
    SUBCASE("Create identity network_with_encoding as shared_ptr") {
        const int encoding_input_width = 64;

        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, encoding_input_width},
                                   {EncodingParams::SCALE, 1.0},
                                   {EncodingParams::OFFSET, 0.0},
                                   {EncodingParams::ENCODING, EncodingNames::IDENTITY}};
        test_create_nwe_as_shared_ptr<float, bf16, 64>(q, encoding_config);
    }
    SUBCASE("Create grid network_with_encoding as shared_ptr") {
        const int encoding_input_width = 3;

        const json encoding_config{{EncodingParams::N_DIMS_TO_ENCODE, encoding_input_width},
                                   {EncodingParams::ENCODING, EncodingNames::GRID},
                                   {EncodingParams::GRID_TYPE, GridType::Hash},
                                   {EncodingParams::N_LEVELS, 16},
                                   {EncodingParams::N_FEATURES_PER_LEVEL, 2},
                                   {EncodingParams::LOG2_HASHMAP_SIZE, 19},
                                   {EncodingParams::BASE_RESOLUTION, 16},
                                   {EncodingParams::PER_LEVEL_SCALE, 2.0}};
        net = test_create_nwe_as_shared_ptr<float, bf16, 64>(q, encoding_config);

        DeviceMatrix<float> params_full_precision(net->get_encoding()->n_params(), 1, q);

        params_full_precision.fill(1.0f).wait();

        net->set_encoding_params(params_full_precision);
    }

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
    SUBCASE("Grid training, random weights, loaded data") {
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
#endif
}

TEST_CASE("Network with Identity Encoding - test inference with padding") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 1;

    auto test_function = [=](auto T_type, sycl::queue &q, const int width) {
        using T = decltype(T_type);

        if (width == 16)
            test_network_with_encoding_identity_inference<16>(q, 16, 16);
        else if (width == 32)
            test_network_with_encoding_identity_inference<32>(q, 32, 32);
        else if (width == 64)
            test_network_with_encoding_identity_inference<64>(q, 64, 64);
        else if (width == 128)
            test_network_with_encoding_identity_inference<128>(q, 128, 128);
        else
            throw std::invalid_argument("Unsupported width");
    };

    const int widths[] = {16, 32, 64, 128};
    auto bf16_type = sycl::ext::oneapi::bfloat16{};
    auto half_type = sycl::half{};

    std::array<decltype(bf16_type), 2> types = {bf16_type, half_type};
    for (auto type : types) {
        std::string type_name = (type == bf16_type) ? "bfloat16" : "half";
        for (int width : widths) {
            std::string testName = "Testing inference " + type_name + " WIDTH " + std::to_string(width);
            SUBCASE(testName.c_str()) { CHECK_NOTHROW(test_function(type, q, width)); }
        }
    }
}
