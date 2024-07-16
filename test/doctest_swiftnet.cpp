/**
 * @file doctest_swiftnet.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the Swiftnet class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"
#include <filesystem>

#include "SwiftNetMLP.h"
#include "io.h"
#include "l2.h"
#include "mlp.h"
#include "result_check.h"
#include "trainer.h"

Activation get_activation_from_string(std::string activation) {
    Activation network_activation;
    if (activation == "relu") {
        network_activation = Activation::ReLU;
    } else if (activation == "sigmoid") {
        network_activation = Activation::Sigmoid;
    } else if (activation == "linear") {
        network_activation = Activation::None;
    }
    return network_activation;
}

template <typename T> std::vector<T> create_padded_vector(int output_width, T target_val, int padded_output_width) {
    if (output_width > padded_output_width) {
        throw std::invalid_argument("output_width cannot be greater than N");
    }

    // Initialize a vector of N zeros
    std::vector<T> target_ref(padded_output_width, 0);
    // Set the first output_width elements to target_val
    std::fill_n(target_ref.begin(), output_width, target_val);
    return target_ref;
}

template <typename T, int WIDTH>
void test_inference_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMatrix<T> network_output(batch_size, network.get_output_width(), q);
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);

    network_input.fill(input_val);

    network.inference(network_input, network_output, {});

    q.wait();

    std::vector<double> out_ref(network_output.size());
    for (int output_idx = 0; output_idx < out_ref.size(); output_idx++) {

        const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
        out_ref[output_idx] =
            nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
    }
    CHECK(areVectorsWithinTolerance(network_output.copy_to_host(), out_ref, 1.0e-2));
}

template <typename T, int WIDTH>
void test_forward_1layer(sycl::queue &q, const int input_width, const int output_width, const int batch_size) {

    constexpr int n_hidden_layers = 1;
    constexpr float input_val = 1.0f;
    // setting Network<T>::WeightInitMode::constant_pos sets the weights to this value
    constexpr float weight_val = 0.01f;

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, Activation::ReLU, Activation::None,
                                  Network<T>::WeightInitMode::constant_pos);

    DeviceMatrices<T> network_interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(),
                                          batch_size, network.get_network_width(), batch_size,
                                          network.get_output_width(), q);
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);
    network_input.fill(input_val);
    q.wait();

    network.forward_pass(network_input, network_interm_forw, {});

    q.wait();

    std::vector<T> fwd_host = network_interm_forw.copy_to_host();

    // input
    CHECK(isVectorWithinTolerance(
        std::vector<T>(fwd_host.begin(), fwd_host.begin() + batch_size * network.get_input_width()), input_val, 1e-3));

    // intermediate
    const double ref_result = weight_val * input_width * input_val;
    CHECK(isVectorWithinTolerance(
        std::vector<T>(fwd_host.begin() + batch_size * network.get_input_width(),
                       fwd_host.begin() + batch_size * (network.get_input_width() + network.get_network_width())),
        ref_result, 1e-3));

    // output
    for (int i = batch_size * (network.get_input_width() + network.get_network_width()); i < fwd_host.size(); i++) {
        const int output_idx = i - batch_size * (network.get_input_width() + network.get_network_width());
        const int nonzero_value = (output_idx % network.get_output_width()) < output_width ? 1 : 0;
        const double ref_result =
            nonzero_value * weight_val * input_width * input_val * network.get_network_width() * weight_val;
        CHECK(static_cast<double>(fwd_host[i]) == doctest::Approx(ref_result).epsilon(1.0e-2));
    }
}

template <typename T, int WIDTH>
void test_grads(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                const int batch_size, std::string activation, std::string output_activation,
                std::string weight_init_mode) {
    T loss_scale = 1.0;
    const int padded_output_width = WIDTH;
    const int padded_input_width = WIDTH;

    Activation network_activation = get_activation_from_string(activation);
    Activation network_output_activation = get_activation_from_string(output_activation);
    const double input_val = 1.0f;
    std::vector<double> input_ref = create_padded_vector<double>(input_width, input_val, padded_input_width);

    const double target_val = 0.1f;
    std::vector<double> target_ref = create_padded_vector<double>(output_width, target_val, padded_output_width);
    mlp_cpp::MLP<double> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation,
                             output_activation, weight_init_mode);
    std::vector<std::vector<double>> fwd_result_ref = mlp.forward(input_ref, false);
    std::vector<double> network_output_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    std::vector<mlp_cpp::Matrix<double>> grad_matrices_ref(n_hidden_layers + 1, mlp_cpp::Matrix<double>(1, 1));
    std::vector<std::vector<double>> loss_grads_ref;
    std::vector<double> loss_ref;
    std::vector<double> dL_dinput_ref;
    std::vector<double> dL_doutput_ref;
    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 loss_scale);

    DeviceMatrix<T> network_output(batch_size, padded_output_width, q);
    network_output.fill(0.0f).wait();
    network_output.copyHostToSubmatrix(mlp_cpp::convert_vector<double, T>(network_output_ref), 0, 0, batch_size,
                                       padded_output_width);

    DeviceMatrix<T> dL_doutput(batch_size, padded_output_width, q);
    dL_doutput.fill(0.0f).wait();

    DeviceMatrix<float> loss(batch_size, padded_output_width, q);
    loss.fill(0.0f).wait();

    DeviceMatrix<float> targets(batch_size, padded_output_width, q);
    targets.fill(0.0f).wait();

    targets.fillSubmatrixWithValue(0, 0, batch_size, output_width, static_cast<float>(target_val));

    L2Loss<T> l2_loss;

    auto output_view = network_output.GetView();
    sycl::event sycl_event = l2_loss.evaluate(q, loss_scale, output_view, targets, loss, dL_doutput);
    q.wait();

    // Calculating backward of swifnet from here using reference values
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, network_activation,
                                  network_output_activation, Network<T>::WeightInitMode::constant_pos);

    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<double, T>(mlp.getUnpackedWeights());
    network.set_weights_matrices(unpacked_weights, false);

    DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(), batch_size,
                                  network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> interm_backw(network.get_n_hidden_layers() + 1, batch_size, network.get_network_width(),
                                   batch_size, network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> grads(network.get_n_hidden_layers() + 1, network.get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                            network.get_output_width(), q);
    grads.fill(0.0f).wait();
    interm_forw.fill((T)0).wait();
    interm_backw.fill((T)0).wait();

    fwd_result_ref = mlp.forward(input_ref, true);
    // in test_loss, test_fwd, we checked that interm_forw and dLdoutput are the same for swiftnet and reference
    std::vector<double> interm_forw_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    // sanity check whether set_weights worked
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);
    network_input.fill((T)0).wait();
    std::vector<T> input_full = mlp_cpp::stack_vector(mlp_cpp::convert_vector<double, T>(input_ref), batch_size);
    network_input.copy_from_host(input_full).wait();

    network.forward_pass(network_input, interm_forw, {});
    q.wait();

    auto interm_forw_vec = interm_forw.copy_to_host();

    if (!areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2)) {
        printVector("interm_forw_vec: ", interm_forw_vec);
        printVector("interm_forw_ref: ", interm_forw_ref);
    }
    CHECK(areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2));

    interm_forw.copy_from_host(mlp_cpp::convert_vector<double, T>(interm_forw_ref)).wait();

    std::vector<double> stacked_dL_doutput_ref = mlp_cpp::stack_vector(dL_doutput_ref, batch_size);

    auto dL_doutput_vec = dL_doutput.copy_to_host();

    bool grads_within_tolerance = areVectorsWithinTolerance(dL_doutput_vec, stacked_dL_doutput_ref, 1.0e-2);
    if (!grads_within_tolerance) {
        printVector("stacked_dL_doutput_ref", stacked_dL_doutput_ref, 0, -1);
        printVector("dL_doutput_vec", dL_doutput_vec, 0, -1);
    }
    CHECK(areVectorsWithinTolerance(dL_doutput_vec, stacked_dL_doutput_ref, 1.0e-2));

    dL_doutput.copy_from_host(mlp_cpp::convert_vector<double, T>(stacked_dL_doutput_ref)).wait();

    // up until here, we load only reference values to test only interm_backw(and grads)
    network.backward_pass(dL_doutput, grads, interm_backw, interm_forw, {});
    q.wait();

    std::vector<T> interm_backw_vec = interm_backw.copy_to_host();
    std::vector<T> grad_vec = grads.copy_to_host();
    q.wait();

    // flatten reference grad matrices
    std::vector<double> grads_ref;
    for (const auto &matrix : grad_matrices_ref) {
        for (size_t row = 0; row < matrix.rows(); ++row) {
            for (size_t col = 0; col < matrix.cols(); ++col) {
                grads_ref.push_back(matrix.data[row][col]);
            }
        }
    }

    std::vector<double> interm_backw_ref;
    for (const auto &inner_vector : loss_grads_ref) {
        auto inner_stacked = mlp_cpp::stack_vector(inner_vector, batch_size);
        for (T value : inner_stacked) {
            interm_backw_ref.push_back(value); // Add each element to the flattened vector
        }
    }

    CHECK(areVectorsWithinTolerance(interm_backw_vec, interm_backw_ref,
                                    1.0e-2)); // sanity check, being tested in test_interm_backw

    if (!areVectorsWithinTolerance(grad_vec, grads_ref, 1.0e-2)) {
        printVector("grads_ref", grads_ref);
        printVector("grad_vec", grad_vec);
    }
    CHECK(areVectorsWithinTolerance(grad_vec, grads_ref, 1.0e-2));
}

template <typename T, int WIDTH>
void test_interm_backw(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                       const int batch_size, std::string activation, std::string output_activation,
                       std::string weight_init_mode) {
    T loss_scale = 1.0;
    const int padded_output_width = WIDTH;
    const int padded_input_width = WIDTH;

    Activation network_activation = get_activation_from_string(activation);
    Activation network_output_activation = get_activation_from_string(output_activation);
    const double input_val = 1.0f;
    std::vector<double> input_ref = create_padded_vector<double>(input_width, input_val, padded_input_width);

    const double target_val = 0.1f;
    std::vector<double> target_ref = create_padded_vector<double>(output_width, target_val, padded_output_width);

    mlp_cpp::MLP<double> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation,
                             output_activation, weight_init_mode);
    std::vector<std::vector<double>> fwd_result_ref = mlp.forward(input_ref, false);
    std::vector<double> network_output_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    std::vector<mlp_cpp::Matrix<double>> grad_matrices_ref(n_hidden_layers + 1, mlp_cpp::Matrix<double>(1, 1));
    std::vector<std::vector<double>> loss_grads_ref;
    std::vector<double> loss_ref;
    std::vector<double> dL_dinput_ref;
    std::vector<double> dL_doutput_ref;
    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 loss_scale);

    DeviceMatrix<T> network_output(batch_size, padded_output_width, q);
    network_output.fill(0.0f).wait();
    network_output.copyHostToSubmatrix(mlp_cpp::convert_vector<double, T>(network_output_ref), 0, 0, batch_size,
                                       padded_output_width);

    DeviceMatrix<T> dL_doutput(batch_size, padded_output_width, q);
    dL_doutput.fill(0.0f).wait();

    DeviceMatrix<float> loss(batch_size, padded_output_width, q);
    loss.fill(0.0f).wait();

    DeviceMatrix<float> targets(batch_size, padded_output_width, q);
    targets.fillSubmatrixWithValue(0, 0, batch_size, output_width, static_cast<float>(target_val));

    L2Loss<T> l2_loss;
    sycl::event sycl_event = l2_loss.evaluate(q, loss_scale, network_output.GetView(), targets, loss, dL_doutput);
    q.wait();

    // Calculating backward of swifnet from here using reference values
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, network_activation,
                                  network_output_activation, Network<T>::WeightInitMode::constant_pos);
    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<double, T>(mlp.getUnpackedWeights());

    network.set_weights_matrices(unpacked_weights, false);
    DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(), batch_size,
                                  network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> interm_backw(network.get_n_hidden_layers() + 1, batch_size, network.get_network_width(),
                                   batch_size, network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> grads(network.get_n_hidden_layers() + 1, network.get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                            network.get_output_width(), q);
    grads.fill(0.0f).wait();
    interm_forw.fill((T)0).wait();
    interm_backw.fill((T)0).wait();

    fwd_result_ref = mlp.forward(input_ref, true);
    // in test_loss, test_fwd, we checked that interm_forw and dLdoutput are the same for swiftnet and reference
    std::vector<double> interm_forw_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    // sanity check whether set_weights worked
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);
    std::vector<T> input_full = mlp_cpp::stack_vector(mlp_cpp::convert_vector<double, T>(input_ref), batch_size);
    network_input.copy_from_host(input_full).wait();

    network.forward_pass(network_input, interm_forw, {});
    q.wait();
    auto interm_forw_vec = interm_forw.copy_to_host();

    if (!areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2)) {
        printVector("interm_forw_vec: ", interm_forw_vec);
        printVector("interm_forw_ref: ", interm_forw_ref);
    }
    CHECK(areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2));

    interm_forw.copy_from_host(mlp_cpp::convert_vector<double, T>(interm_forw_ref)).wait();

    std::vector<double> stacked_dL_doutput_ref = mlp_cpp::stack_vector(dL_doutput_ref, batch_size);
    dL_doutput.copy_from_host(mlp_cpp::convert_vector<double, T>(stacked_dL_doutput_ref)).wait();

    // up until here, we load only reference values to test only interm_backw(and grads)
    network.backward_pass(dL_doutput, grads, interm_backw, interm_forw, {});
    q.wait();

    std::vector<T> interm_backw_vec = interm_backw.copy_to_host();
    q.wait();

    for (int i = 0; i < loss_grads_ref.size(); i++) {
        std::vector<double> interm_backw_ref;
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
}

template <typename T, int WIDTH>
void test_dl_dinput(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                    const int batch_size, std::string activation, std::string output_activation,
                    std::string weight_init_mode) {
    T loss_scale = 1.0;
    const int padded_output_width = WIDTH;
    const int padded_input_width = WIDTH;

    Activation network_activation = get_activation_from_string(activation);
    Activation network_output_activation = get_activation_from_string(output_activation);
    const double input_val = 1.0f;
    std::vector<double> input_ref = create_padded_vector<double>(input_width, input_val, padded_input_width);
    const double target_val = 0.1f;
    std::vector<double> target_ref = create_padded_vector<double>(output_width, target_val, padded_output_width);

    mlp_cpp::MLP<double> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation,
                             output_activation, weight_init_mode);
    std::vector<std::vector<double>> fwd_result_ref = mlp.forward(input_ref, false);
    std::vector<double> network_output_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    std::vector<mlp_cpp::Matrix<double>> grad_matrices_ref(n_hidden_layers + 1, mlp_cpp::Matrix<double>(1, 1));
    std::vector<std::vector<double>> loss_grads_ref;
    std::vector<double> loss_ref;
    std::vector<double> dL_dinput_ref;
    std::vector<double> dL_doutput_ref;
    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 loss_scale);

    DeviceMatrix<T> network_output(batch_size, padded_output_width, q);
    network_output.fill(0.0f).wait();
    network_output.copyHostToSubmatrix(mlp_cpp::convert_vector<double, T>(network_output_ref), 0, 0, batch_size,
                                       padded_output_width);

    DeviceMatrix<T> dL_doutput(batch_size, padded_output_width, q);
    dL_doutput.fill(0.0f).wait();

    DeviceMatrix<float> loss(batch_size, padded_output_width, q);
    loss.fill(0.0f).wait();

    DeviceMatrix<float> targets(batch_size, padded_output_width, q);
    targets.fillSubmatrixWithValue(0, 0, batch_size, output_width, static_cast<float>(target_val));

    DeviceMatrix<T> dL_dinput(batch_size, input_width, q);
    dL_dinput.fill(0.123f).wait();
    L2Loss<T> l2_loss;
    sycl::event sycl_event = l2_loss.evaluate(q, loss_scale, network_output.GetView(), targets, loss, dL_doutput);
    q.wait();

    // Calculating backward of swifnet from here using reference values
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, network_activation,
                                  network_output_activation, Network<T>::WeightInitMode::constant_pos);
    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<double, T>(mlp.getUnpackedWeights());
    network.set_weights_matrices(unpacked_weights, false);
    DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(), batch_size,
                                  network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> interm_backw(network.get_n_hidden_layers() + 1, batch_size, network.get_network_width(),
                                   batch_size, network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> grads(network.get_n_hidden_layers() + 1, network.get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                            network.get_output_width(), q);
    grads.fill(0.0f).wait();
    interm_forw.fill((T)0).wait();
    interm_backw.fill((T)0).wait();

    fwd_result_ref = mlp.forward(input_ref, true);
    // in test_loss, test_fwd, we checked that interm_forw and dLdoutput are the same for swiftnet and reference
    std::vector<double> interm_forw_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    // sanity check whether set_weights worked
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);
    std::vector<T> input_full = mlp_cpp::stack_vector(mlp_cpp::convert_vector<double, T>(input_ref), batch_size);
    network_input.copy_from_host(input_full).wait();

    network.forward_pass(network_input, interm_forw, {});
    q.wait();
    auto interm_forw_vec = interm_forw.copy_to_host();

    if (!areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2)) {
        printVector("interm_forw_vec: ", interm_forw_vec, batch_size * WIDTH, -1);
        printVector("interm_forw_ref: ", interm_forw_ref, batch_size * WIDTH, -1);
    }
    CHECK(areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2));

    interm_forw.copy_from_host(mlp_cpp::convert_vector<double, T>(interm_forw_ref)).wait();

    std::vector<double> stacked_dL_doutput_ref = mlp_cpp::stack_vector(dL_doutput_ref, batch_size);
    dL_doutput.copy_from_host(mlp_cpp::convert_vector<double, T>(stacked_dL_doutput_ref)).wait();

    // up until here, we load only reference values to test only interm_backw(and grads)
    auto dL_dinput_view = dL_dinput.GetView();
    network.backward_pass(dL_doutput, grads, interm_backw, interm_forw, {}, &dL_dinput_view);
    q.wait();

    std::vector<T> dL_dinput_vec = dL_dinput.copy_to_host();
    q.wait();

    auto dL_dinput_ref_stacked = mlp_cpp::stack_vector(dL_dinput_ref, batch_size);

    if (!areVectorsWithinTolerance(dL_dinput_vec, dL_dinput_ref_stacked, 1.0e-2)) {
        printVector("dL_dinput_ref_stacked: ", dL_dinput_ref_stacked);
        printVector("dL_dinput_vec: ", dL_dinput_vec);
    }
    CHECK(areVectorsWithinTolerance(dL_dinput_vec, dL_dinput_ref_stacked, 1.0e-2));
}

template <typename T, int WIDTH>
void test_loss(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
               const int batch_size, std::string activation, std::string output_activation,
               std::string weight_init_mode) {
    T loss_scale = 10.0; // randomly picking one to ensure it works

    const int padded_output_width = WIDTH;
    const int padded_input_width = WIDTH;

    const double input_val = 1.0f;
    std::vector<double> input_ref = create_padded_vector<double>(input_width, input_val, padded_input_width);

    const double target_val = 0.1f;
    std::vector<double> target_ref = create_padded_vector<double>(output_width, target_val, padded_output_width);

    mlp_cpp::MLP<double> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation,
                             output_activation, weight_init_mode);

    std::vector<std::vector<double>> fwd_result_ref = mlp.forward(input_ref, false);
    std::vector<double> network_output_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    std::vector<mlp_cpp::Matrix<double>> weights_ref(n_hidden_layers + 1, mlp_cpp::Matrix<double>(1, 1));
    std::vector<std::vector<double>> loss_grads_ref;
    std::vector<double> loss_ref;
    std::vector<double> dL_doutput_ref;
    std::vector<double> dL_dinput_ref;

    mlp.backward(input_ref, target_ref, weights_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 loss_scale);

    DeviceMatrix<T> network_output(batch_size, padded_output_width, q);
    network_output.fill(0.0f).wait();
    network_output.copyHostToSubmatrix(mlp_cpp::convert_vector<double, T>(network_output_ref), 0, 0, batch_size,
                                       padded_output_width);
    q.wait();
    DeviceMatrix<T> dL_doutput(batch_size, padded_output_width, q);
    dL_doutput.fill(0.0f).wait();

    DeviceMatrix<float> loss(batch_size, padded_output_width, q);
    loss.fill(0.0f).wait();

    DeviceMatrix<float> targets(batch_size, padded_output_width, q);
    targets.fill(0.0f).wait();
    targets.fillSubmatrixWithValue(0, 0, batch_size, output_width, target_val);
    q.wait();

    L2Loss<T> l2_loss;
    sycl::event sycl_event = l2_loss.evaluate(q, loss_scale, network_output.GetView(), targets, loss, dL_doutput);
    q.wait();

    std::vector<T> dL_doutput_vec = dL_doutput.copy_to_host();
    std::vector<double> stacked_dL_doutput_ref = mlp_cpp::stack_vector(dL_doutput_ref, batch_size);

    if (!areVectorsWithinTolerance(dL_doutput_vec, stacked_dL_doutput_ref, 1.0e-2)) {
        printVector("stacked_dL_doutput_ref", stacked_dL_doutput_ref);
        printVector("dL_doutput_vec", dL_doutput_vec);
    }
    CHECK(areVectorsWithinTolerance(dL_doutput_vec, stacked_dL_doutput_ref, 1.0e-2));
    std::vector<double> stacked_loss_ref = mlp_cpp::stack_vector(loss_ref, batch_size);
    auto loss_vec = loss.copy_to_host();

    if (!areVectorsWithinTolerance(loss_vec, stacked_loss_ref, 1.0e-2)) {
        printVector("stacked_loss_ref", stacked_loss_ref);
        printVector("loss_vec", loss_vec);
    }
    CHECK(areVectorsWithinTolerance(loss_vec, stacked_loss_ref, 1.0e-2));
}

template <typename T, int WIDTH>
void test_interm_fwd(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                     const int batch_size, std::string activation, std::string output_activation,
                     std::string weight_init, bool random_input) {
    CHECK(input_width ==
          WIDTH); // if want to remove this, simply adapt the script to have padded_input_width (see e.g., test_grad())
    CHECK(output_width == WIDTH);
    Activation network_activation = get_activation_from_string(activation);
    Activation network_output_activation = get_activation_from_string(output_activation);
    const double input_val = 1.0f;
    std::vector<double> input_ref(input_width);
    if (random_input) {
        std::mt19937 gen(42);
        // Create a distribution in the range [input_min, input_max]
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        // Fill the vector with random numbers
        for (double &value : input_ref) {
            value = dis(gen);
        }
    } else {
        for (double &value : input_ref) {
            value = input_val;
        }
    }

    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, network_activation,
                                  network_output_activation, Network<T>::WeightInitMode::constant_pos);

    mlp_cpp::MLP<double> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation,
                             output_activation, weight_init);

    std::vector<double> unpacked_weights_double = mlp.getUnpackedWeights();

    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<double, T>(unpacked_weights_double);
    network.set_weights_matrices(unpacked_weights, false);

    DeviceMatrix<T> network_input(batch_size, input_width, q);
    DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(), batch_size,
                                  network.get_network_width(), batch_size, network.get_output_width(), q);
    // Repeat source vector N times
    std::vector<T> input_full = mlp_cpp::stack_vector(mlp_cpp::convert_vector<double, T>(input_ref), batch_size);

    network_input.copy_from_host(input_full).wait();

    interm_forw.fill((T)0).wait();

    network.forward_pass(network_input, interm_forw, {});
    q.wait();

    std::vector<std::vector<double>> fwd_result_ref = mlp.forward(input_ref, true);

    auto interm_forw_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);
    auto interm_forw_vec = interm_forw.copy_to_host();
    if (!areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2)) {
        printVector("interm_forw_vec", interm_forw_vec, WIDTH * batch_size);
        printVector("interm_forw_ref", interm_forw_ref, WIDTH * batch_size);
    }

    CHECK(interm_forw_vec.size() == interm_forw_ref.size());
    CHECK(areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref, 1.0e-2));
}

template <typename T, int WIDTH>
void test_trainer(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                  const int batch_size, std::string activation, std::string output_activation,
                  std::string weight_init_mode) {
    T loss_scale = 1.0;
    const int padded_output_width = WIDTH;
    const int padded_input_width = WIDTH;

    Activation network_activation = get_activation_from_string(activation);
    Activation network_output_activation = get_activation_from_string(output_activation);
    const double input_val = 1.0f;
    std::vector<double> input_ref = create_padded_vector<double>(input_width, input_val, padded_input_width);

    const double target_val = 0.1f;
    std::vector<double> target_ref = create_padded_vector<double>(output_width, target_val, padded_output_width);

    mlp_cpp::MLP<double> mlp(input_width, WIDTH, output_width, n_hidden_layers + 1, batch_size, activation,
                             output_activation, weight_init_mode);
    std::vector<std::vector<double>> fwd_result_ref = mlp.forward(input_ref, false);
    std::vector<double> network_output_ref = mlp_cpp::repeat_inner_vectors<double>(fwd_result_ref, batch_size);

    std::vector<mlp_cpp::Matrix<double>> grad_matrices_ref(n_hidden_layers + 1, mlp_cpp::Matrix<double>(1, 1));
    std::vector<std::vector<double>> loss_grads_ref;
    std::vector<double> loss_ref;
    std::vector<double> dL_dinput_ref;
    std::vector<double> dL_doutput_ref;
    mlp.backward(input_ref, target_ref, grad_matrices_ref, loss_grads_ref, loss_ref, dL_doutput_ref, dL_dinput_ref,
                 loss_scale);

    // Calculating backward of swifnet from here using reference values
    SwiftNetMLP<T, WIDTH> network(q, input_width, output_width, n_hidden_layers, network_activation,
                                  network_output_activation, Network<T>::WeightInitMode::constant_pos);

    std::vector<T> unpacked_weights = mlp_cpp::convert_vector<double, T>(mlp.getUnpackedWeights());
    network.set_weights_matrices(unpacked_weights, false);

    DeviceMatrices<T> interm_forw(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(), batch_size,
                                  network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> interm_backw(network.get_n_hidden_layers() + 1, batch_size, network.get_network_width(),
                                   batch_size, network.get_network_width(), batch_size, network.get_output_width(), q);
    DeviceMatrices<T> grads(network.get_n_hidden_layers() + 1, network.get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                            network.get_output_width(), q);
    grads.fill(0.0f).wait();
    interm_forw.fill((T)0).wait();
    interm_backw.fill((T)0).wait();
    DeviceMatrix<T> dL_doutput(batch_size, padded_output_width, q);
    dL_doutput.fill(0.0f).wait();

    // sanity check whether set_weights worked
    DeviceMatrix<T> network_input(batch_size, network.get_input_width(), q);
    std::vector<T> input_full = mlp_cpp::stack_vector(mlp_cpp::convert_vector<double, T>(input_ref), batch_size);
    network_input.copy_from_host(input_full).wait();
    std::vector<double> stacked_dL_doutput_ref = mlp_cpp::stack_vector(dL_doutput_ref, batch_size);
    dL_doutput.copy_from_host(mlp_cpp::convert_vector<double, T>(stacked_dL_doutput_ref)).wait();

    // warm up and set the baseline
    DeviceMatrices<T> grads_ref(network.get_n_hidden_layers() + 1, network.get_network_width(), WIDTH, WIDTH, WIDTH,
                                WIDTH, network.get_output_width(), q);
    DeviceMatrices<T> interm_forw_ref(network.get_n_hidden_layers() + 2, batch_size, network.get_input_width(),
                                      batch_size, network.get_network_width(), batch_size, network.get_output_width(),
                                      q);
    DeviceMatrices<T> interm_backw_ref(network.get_n_hidden_layers() + 1, batch_size, network.get_network_width(),
                                       batch_size, network.get_network_width(), batch_size, network.get_output_width(),
                                       q);
    auto deps = network.forward_pass(network_input, interm_forw_ref, {});
    network.backward_pass(dL_doutput, grads_ref, interm_backw_ref, interm_forw_ref, deps);
    q.wait();

    Trainer<T> trainer{&network};
    // training step is recorded in a sycl graph
    trainer.training_step(network_input, grads, dL_doutput, interm_forw, interm_backw, {});

    auto interm_forw_vec = interm_forw.copy_to_host();
    auto interm_forw_ref_vec = interm_forw_ref.copy_to_host();
    CHECK(areVectorsWithinTolerance(interm_forw_vec, interm_forw_ref_vec, 1.0e-2));

    auto interm_backw_vec = interm_backw.copy_to_host();
    auto interm_backw_ref_vec = interm_backw_ref.copy_to_host();
    CHECK(areVectorsWithinTolerance(interm_backw_vec, interm_backw_ref_vec, 1.0e-2));

    auto grad_vec = grads.copy_to_host();
    auto grad_ref_vec = grads_ref.copy_to_host();
    CHECK(areVectorsWithinTolerance(grad_vec, grad_ref_vec, 1.0e-2));
}

// TEST_CASE("Swiftnet - Constructor") {

//     sycl::queue q;
//     typedef sycl::ext::oneapi::bfloat16 T;

//     // No need to test width template parameter since it is statically asserted in swiftnetmlp class No need to test
//     // type template parameter since it is statically asserted in Network class
//     SUBCASE("Supported 1") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Supported 2") { CHECK_NOTHROW(SwiftNetMLP<T, 32>(q, 32, 32, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Supported 3") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 64, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Supported 4") { CHECK_NOTHROW(SwiftNetMLP<T, 128>(q, 128, 128, 4, Activation::ReLU, Activation::None)); }

//     SUBCASE("Supported 5") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::None, Activation::None)); }
//     SUBCASE("Supported 6") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::None, Activation::ReLU)); }
//     SUBCASE("Supported 7") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::None, Activation::Sigmoid)); }

//     SUBCASE("Supported 8") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Supported 9") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::ReLU, Activation::ReLU)); }
//     SUBCASE("Supported 10") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::ReLU, Activation::Sigmoid)); }

//     SUBCASE("Supported 11") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Sigmoid, Activation::None)); }
//     SUBCASE("Supported 12") { CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Sigmoid, Activation::ReLU)); }
//     SUBCASE("Supported 13") {
//         CHECK_NOTHROW(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Sigmoid, Activation::Sigmoid));
//     }

//     SUBCASE("Pad input 1") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 16, 64, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Pad input 2") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 1, 64, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Pad output 1") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 1, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Pad output 2") { CHECK_NOTHROW(SwiftNetMLP<T, 64>(q, 64, 16, 4, Activation::ReLU, Activation::None)); }
//     SUBCASE("Unsupported layers 1") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 0, Activation::ReLU, Activation::None));
//     }
//     SUBCASE("Unsupported layers 2") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, -1, Activation::ReLU, Activation::None));
//     }
//     SUBCASE("Unsupported input width 1") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, -1, 16, 4, Activation::ReLU, Activation::None));
//     }
//     SUBCASE("Unsupported output width 1") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, -1, 4, Activation::ReLU, Activation::None));
//     }
//     SUBCASE("Unsupported activation 1") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Tanh, Activation::None));
//     }
//     SUBCASE("Unsupported activation 2") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Tanh, Activation::ReLU));
//     }
//     SUBCASE("Unsupported activation 3") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Tanh, Activation::Sigmoid));
//     }
//     SUBCASE("Unsupported output activation 1") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::None, Activation::Tanh));
//     }
//     SUBCASE("Unsupported output activation 2") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::ReLU, Activation::Tanh));
//     }
//     SUBCASE("Unsupported output activation 3") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Sigmoid, Activation::Tanh));
//     }
//     SUBCASE("Unsupported activation and output activation") {
//         CHECK_THROWS(SwiftNetMLP<T, 16>(q, 16, 16, 4, Activation::Tanh, Activation::Tanh));
//     }
// }

// /// TODO: check if the weights are actually 0 whereever they should be.
// TEST_CASE("Swiftnet - Zero Padding") {
//     sycl::queue q;
//     typedef sycl::ext::oneapi::bfloat16 T;
//     SUBCASE("Input 1-64") {
//         SwiftNetMLP<T, 64> network(q, 1, 64, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 64);
//         CHECK(network.get_network_width() == 64);
//         CHECK(network.get_output_width() == 64);
//     }
//     SUBCASE("Input 1-16") {
//         SwiftNetMLP<T, 16> network(q, 1, 16, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 16);
//         CHECK(network.get_network_width() == 16);
//         CHECK(network.get_output_width() == 16);
//     }
//     SUBCASE("Input 17-32") {
//         SwiftNetMLP<T, 32> network(q, 17, 32, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 32);
//         CHECK(network.get_network_width() == 32);
//         CHECK(network.get_output_width() == 32);
//     }
//     SUBCASE("Input 17-128") {
//         SwiftNetMLP<T, 128> network(q, 17, 128, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 128);
//         CHECK(network.get_network_width() == 128);
//         CHECK(network.get_output_width() == 128);
//     }
//     SUBCASE("Output 1-64") {
//         SwiftNetMLP<T, 64> network(q, 64, 1, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 64);
//         CHECK(network.get_network_width() == 64);
//         CHECK(network.get_output_width() == 64);
//     }
//     SUBCASE("Output 1-16") {
//         SwiftNetMLP<T, 16> network(q, 16, 1, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 16);
//         CHECK(network.get_network_width() == 16);
//         CHECK(network.get_output_width() == 16);
//     }
//     SUBCASE("Output 17-32") {
//         SwiftNetMLP<T, 32> network(q, 32, 17, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 32);
//         CHECK(network.get_network_width() == 32);
//         CHECK(network.get_output_width() == 32);
//     }
//     SUBCASE("Output 17-128") {
//         SwiftNetMLP<T, 128> network(q, 128, 17, 4, Activation::ReLU, Activation::None);
//         CHECK(network.get_input_width() == 128);
//         CHECK(network.get_network_width() == 128);
//         CHECK(network.get_output_width() == 128);
//     }
// }

// TEST_CASE("Swiftnet - weights init") {
//     sycl::queue q(sycl::gpu_selector_v);
//     typedef sycl::ext::oneapi::bfloat16 T;

//     SUBCASE("Default positive, No Pad") {
//         SwiftNetMLP<T, 64> network(q, 64, 64, 4, Activation::ReLU, Activation::None,
//                                    Network<T>::WeightInitMode::constant_pos);
//         CHECK_NOTHROW(network.get_weights_matrices());
//         CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);
//         for (int iter = 0; iter < 5; iter++) {
//             CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
//             CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
//         }

//         CHECK(areVectorsWithinTolerance(network.get_weights_matrices().copy_to_host(),
//                                         std::vector<T>(network.get_weights_matrices().nelements(), 0.01), 1e-3));
//     }

//     SUBCASE("Default positive, Output Pad") {
//         SwiftNetMLP<T, 64> network(q, 64, 63, 4, Activation::ReLU, Activation::None,
//                                    Network<T>::WeightInitMode::constant_pos);
//         CHECK_NOTHROW(network.get_weights_matrices());
//         CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);

//         for (int iter = 0; iter < 4; iter++) {
//             CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
//             CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
//         }
//         CHECK(network.get_weights_matrices().Back().m() == 64);
//         CHECK(network.get_weights_matrices().Back().n() == 64);
//     }

//     SUBCASE("Overwrite, No Pad") {
//         SwiftNetMLP<T, 64> network(q, 64, 64, 4, Activation::ReLU, Activation::None,
//                                    Network<T>::WeightInitMode::constant_pos);
//         CHECK_NOTHROW(network.get_weights_matrices());
//         CHECK(network.get_weights_matrices().GetNumberOfMatrices() == 5);
//         std::vector<T> new_weights(network.get_weights_matrices().nelements(), 1.23);
//         network.set_weights_matrices(new_weights, true);
//         for (int iter = 0; iter < 5; iter++) {
//             CHECK(network.get_weights_matrices().GetView(iter).m() == 64);
//             CHECK(network.get_weights_matrices().GetView(iter).n() == 64);
//         }

//         CHECK(areVectorsWithinTolerance(network.get_weights_matrices().copy_to_host(),
//                                         std::vector<T>(network.get_weights_matrices().nelements(), 1.23), 1e-3));
//     }
// }

// TEST_CASE("Swiftnet - zero pad forward_pass WIDTH 64") {
//     sycl::queue q(sycl::gpu_selector_v);

//     auto test_function = [=](const int input_width, const int output_width, sycl::queue &q) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         constexpr int WIDTH = 64;
//         test_forward_1layer<T, WIDTH>(q, input_width, output_width, 8);
//     };

//     SUBCASE("No Pad") {
//         constexpr int input_width = 64;
//         constexpr int output_width = 64;
//         test_function(input_width, output_width, q);
//     }
//     SUBCASE("Input Pad") {
//         constexpr int input_width = 3;
//         constexpr int output_width = 64;
//         test_function(input_width, output_width, q);
//     }
//     SUBCASE("Output Pad") {
//         constexpr int input_width = 64;
//         constexpr int output_width = 7;
//         test_function(input_width, output_width, q);
//     }
//     SUBCASE("Input and Output Pad") {
//         constexpr int input_width = 3;
//         constexpr int output_width = 5;
//         test_function(input_width, output_width, q);
//     }
// }

// TEST_CASE("Swiftnet - zero pad inference WIDTH 64") {
//     sycl::queue q(sycl::gpu_selector_v);

//     auto test_function = [=](const int input_width, const int output_width, sycl::queue &q) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         constexpr int WIDTH = 64;
//         test_inference_1layer<T, WIDTH>(q, input_width, output_width, 8);
//     };

//     SUBCASE("No Pad") {
//         constexpr int input_width = 64;
//         constexpr int output_width = 64;
//         test_function(input_width, output_width, q);
//     }
//     SUBCASE("Input Pad") {
//         constexpr int input_width = 3;
//         constexpr int output_width = 64;
//         test_function(input_width, output_width, q);
//     }
//     SUBCASE("Output Pad") {
//         constexpr int input_width = 64;
//         constexpr int output_width = 7;
//         test_function(input_width, output_width, q);
//     }
//     SUBCASE("Input and Output Pad") {
//         constexpr int input_width = 3;
//         constexpr int output_width = 5;
//         test_function(input_width, output_width, q);
//     }
// }

// TEST_CASE("Swiftnet - Batch Sizes forward") {
//     sycl::queue q(sycl::gpu_selector_v);

//     auto test_function = [=](const int batch_size, sycl::queue &q) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         constexpr int WIDTH = 64;
//         test_forward_1layer<T, WIDTH>(q, WIDTH, WIDTH, batch_size);
//     };

//     SUBCASE("Batch size 8") { CHECK_NOTHROW(test_function(8, q)); }
//     SUBCASE("Batch size 512") { CHECK_NOTHROW(test_function(512, q)); }
//     SUBCASE("Batch size 16") { CHECK_NOTHROW(test_function(16, q)); }
//     SUBCASE("Batch size 1") { CHECK_THROWS(test_function(1, q)); }
//     SUBCASE("Batch size 13") { CHECK_THROWS(test_function(13, q)); }
// }

// TEST_CASE("Swiftnet - Net Widths forward") {
//     // only testing constructor. values tested later
//     sycl::queue q(sycl::gpu_selector_v);

//     auto test_function = [=](const int width, sycl::queue &q) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         if (width == 16)
//             test_forward_1layer<T, 16>(q, 16, 16, 8);
//         else if (width == 32)
//             test_forward_1layer<T, 32>(q, 32, 32, 8);
//         else if (width == 64)
//             test_forward_1layer<T, 64>(q, 64, 64, 8);
//         else if (width == 128)
//             test_forward_1layer<T, 128>(q, 128, 128, 8);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };

//     SUBCASE("WIDTH 16") { CHECK_NOTHROW(test_function(16, q)); }
//     SUBCASE("WIDTH 32") { CHECK_NOTHROW(test_function(32, q)); }
//     SUBCASE("WIDTH 64") { CHECK_NOTHROW(test_function(64, q)); }
//     SUBCASE("WIDTH 128") { CHECK_NOTHROW(test_function(128, q)); }
// }

// TEST_CASE("Swiftnet - test interm_fwd with reference MLP") {

//     // only testing constructor. values tested later
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 1;
//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init, bool random_input) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         if (width == 16)
//             test_interm_fwd<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation, output_activation, weight_init,
//                                    random_input);
//         else if (width == 32)
//             test_interm_fwd<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation, output_activation, weight_init,
//                                    random_input);
//         else if (width == 64)
//             test_interm_fwd<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation, output_activation, weight_init,
//                                    random_input);
//         else if (width == 128)
//             test_interm_fwd<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation, output_activation,
//                                     weight_init, random_input);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid", "relu"};
//     std::string weight_inits[] = {"linear", "sigmoid", "relu"};
//     bool random_inputs[] = {true, false};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init : weight_inits) {
//                         for (bool random_input : random_inputs) {
//                             std::string random_string = random_input ? "true" : "false";
//                             std::string testName =
//                                 "Testing interm_fwd WIDTH " + std::to_string(width) + " - activation: " + activation +
//                                 " - output_activation: " + output_activation +
//                                 " - Batch size: " + std::to_string(batch_size) + " - weight init: " + weight_init +
//                                 " - random input:" + random_string;
//                             SUBCASE(testName.c_str()) {
//                                 CHECK_NOTHROW(test_function(q, width, batch_size, activation, output_activation,
//                                                             weight_init, random_input));
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST_CASE("Swiftnet - test loss") {
//     // only testing constructor. values tested later
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 1;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef float T; // double is ok too, but not supported on arc
//         if (width == 16)
//             test_loss<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation, output_activation, weight_init_mode);
//         else if (width == 32)
//             test_loss<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation, output_activation, weight_init_mode);
//         else if (width == 64)
//             test_loss<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation, output_activation, weight_init_mode);
//         else if (width == 128)
//             test_loss<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation, output_activation,
//                               weight_init_mode);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid", "relu"};
//     std::string weight_init_modes[] = {"constant", "random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing loss WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode +
//                             " - Batch size : " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST_CASE("Swiftnet - test interm bwd") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 2;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         if (width == 16)
//             test_interm_backw<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation, output_activation,
//                                      weight_init_mode);
//         else if (width == 32)
//             test_interm_backw<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation, output_activation,
//                                      weight_init_mode);
//         else if (width == 64)
//             test_interm_backw<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation, output_activation,
//                                      weight_init_mode);
//         else if (width == 128)
//             test_interm_backw<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation, output_activation,
//                                       weight_init_mode);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid", "relu"};
//     std::string weight_init_modes[] = {"constant", "random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing interm bwd WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode +
//                             " - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

TEST_CASE("Swiftnet - test dL_dinput") {
    sycl::queue q(sycl::gpu_selector_v);
    const int n_hidden_layers = 2;
            test_dl_dinput<sycl::ext::oneapi::bfloat16 , 16>(q, 16, 16, n_hidden_layers, 8, "linear", "relu",
                                  "random");
    // auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
    //                          std::string output_activation, std::string weight_init_mode) {
    //     typedef sycl::ext::oneapi::bfloat16 T;
    //     if (width == 16)
    //         test_dl_dinput<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation, output_activation,
    //                               weight_init_mode);
    //     else if (width == 32)
    //         test_dl_dinput<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation, output_activation,
    //                               weight_init_mode);
    //     else if (width == 64)
    //         test_dl_dinput<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation, output_activation,
    //                               weight_init_mode);
    //     else if (width == 128)
    //         test_dl_dinput<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation, output_activation,
    //                                weight_init_mode);
    //     else
    //         throw std::invalid_argument("Unsupported width");
    // };
    // const int widths[] = {16, 32, 64, 128};
    // const int batch_sizes[] = {8, 16, 32, 64};
    // std::string activations[] = {"linear", "sigmoid", "relu"};
    // std::string output_activations[] = {"linear", "sigmoid", "relu"};
    // std::string weight_init_modes[] = {"constant", "random"};

    // for (int batch_size : batch_sizes) {
    //     for (int width : widths) {
    //         for (std::string activation : activations) {
    //             for (std::string output_activation : output_activations) {
    //                 for (std::string weight_init_mode : weight_init_modes) {
    //                     std::string testName =
    //                         "Testing interm bwd WIDTH " + std::to_string(width) + " - activation: " + activation +
    //                         " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode +
    //                         " - Batch size: " + std::to_string(batch_size);
    //                     SUBCASE(testName.c_str()) {
    //                         CHECK_NOTHROW(
    //                             test_function(q, width, batch_size, activation, output_activation, weight_init_mode));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}
// TEST_CASE("Swiftnet - test interm bwd padded") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 2;
//     const int output_width = 4;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         if (width == 16)
//             test_interm_backw<T, 16>(q, 16, output_width, n_hidden_layers, batch_size, activation, output_activation,
//                                      weight_init_mode);
//         else if (width == 32)
//             test_interm_backw<T, 32>(q, 32, output_width, n_hidden_layers, batch_size, activation, output_activation,
//                                      weight_init_mode);
//         else if (width == 64)
//             test_interm_backw<T, 64>(q, 64, output_width, n_hidden_layers, batch_size, activation, output_activation,
//                                      weight_init_mode);
//         else if (width == 128)
//             test_interm_backw<T, 128>(q, 128, output_width, n_hidden_layers, batch_size, activation, output_activation,
//                                       weight_init_mode);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid", "relu"};
//     std::string weight_init_modes[] = {"constant", "random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing interm bwd WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode +
//                             " - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST_CASE("Swiftnet - test grad unpadded") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 2;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         if (width == 16)
//             test_grads<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation, output_activation, weight_init_mode);
//         else if (width == 32)
//             test_grads<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation, output_activation, weight_init_mode);
//         else if (width == 64)
//             test_grads<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation, output_activation, weight_init_mode);
//         else if (width == 128)
//             test_grads<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation, output_activation,
//                                weight_init_mode);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid", "relu"};
//     std::string weight_init_modes[] = {"constant", "random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing grad WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode +
//                             " - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST_CASE("Swiftnet - test grad output padded") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 2;
//     const int output_dim = 8;
//     int batch_size = 8;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         if (width == 16)
//             test_grads<T, 16>(q, 16, output_dim, n_hidden_layers, batch_size, activation, output_activation,
//                               weight_init_mode);
//         else if (width == 32)
//             test_grads<T, 32>(q, 32, output_dim, n_hidden_layers, batch_size, activation, output_activation,
//                               weight_init_mode);
//         else if (width == 64)
//             test_grads<T, 64>(q, 64, output_dim, n_hidden_layers, batch_size, activation, output_activation,
//                               weight_init_mode);
//         else if (width == 128)
//             test_grads<T, 128>(q, 128, output_dim, n_hidden_layers, batch_size, activation, output_activation,
//                                weight_init_mode);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64, 1 << 17};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid"};
//     std::string weight_init_modes[] = {"random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing grad WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode +
//                             " - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST_CASE("Swiftnet - test grad input padded") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 2;
//     const int input_dim = 8;
//     int batch_size = 8;

//     auto test_function = [=](sycl::queue &q, const int width, const int batch_size, std::string activation,
//                              std::string output_activation, std::string weight_init_mode) {
//         typedef sycl::ext::oneapi::bfloat16 T;
//         if (width == 16)
//             test_grads<T, 16>(q, input_dim, 16, n_hidden_layers, batch_size, activation, output_activation,
//                               weight_init_mode);
//         else if (width == 32)
//             test_grads<T, 32>(q, input_dim, 32, n_hidden_layers, batch_size, activation, output_activation,
//                               weight_init_mode);
//         else if (width == 64)
//             test_grads<T, 64>(q, input_dim, 64, n_hidden_layers, batch_size, activation, output_activation,
//                               weight_init_mode);
//         else if (width == 128)
//             test_grads<T, 128>(q, input_dim, 128, n_hidden_layers, batch_size, activation, output_activation,
//                                weight_init_mode);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64, 1 << 17};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid"};
//     std::string weight_init_modes[] = {"random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName =
//                             "Testing grad WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode +
//                             " - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName.c_str()) {
//                             CHECK_NOTHROW(
//                                 test_function(q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST_CASE("Swiftnet - test trainer") {
//     sycl::queue q(sycl::gpu_selector_v);
//     const int n_hidden_layers = 2;

//     auto test_function = [=]<typename T>(sycl::queue &q, const int width, const int batch_size, std::string
//     activation,
//                                          std::string output_activation, std::string weight_init_mode) {
//         if (width == 16)
//             test_trainer<T, 16>(q, 16, 16, n_hidden_layers, batch_size, activation, output_activation,
//                                 weight_init_mode);
//         else if (width == 32)
//             test_trainer<T, 32>(q, 32, 32, n_hidden_layers, batch_size, activation, output_activation,
//                                 weight_init_mode);
//         else if (width == 64)
//             test_trainer<T, 64>(q, 64, 64, n_hidden_layers, batch_size, activation, output_activation,
//                                 weight_init_mode);
//         else if (width == 128)
//             test_trainer<T, 128>(q, 128, 128, n_hidden_layers, batch_size, activation, output_activation,
//                                  weight_init_mode);
//         else
//             throw std::invalid_argument("Unsupported width");
//     };
//     const int widths[] = {16, 32, 64, 128};
//     const int batch_sizes[] = {8, 16, 32, 64, 1 << 17};
//     std::string activations[] = {"linear", "sigmoid", "relu"};
//     std::string output_activations[] = {"linear", "sigmoid", "relu"};
//     std::string weight_init_modes[] = {"constant", "random"};

//     for (int batch_size : batch_sizes) {
//         for (int width : widths) {
//             for (std::string activation : activations) {
//                 for (std::string output_activation : output_activations) {
//                     for (std::string weight_init_mode : weight_init_modes) {
//                         std::string testName_bf16 =
//                             "Testing bf16 WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode
//                             + " - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName_bf16.c_str()) {
//                             CHECK_NOTHROW(test_function.template operator()<sycl::ext::oneapi::bfloat16>(
//                                 q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }

//                         std::string testName_half =
//                             "Testing half WIDTH " + std::to_string(width) + " - activation: " + activation +
//                             " - output_activation: " + output_activation + " - weight_init_mode: " + weight_init_mode
//                             + " - Batch size: " + std::to_string(batch_size);
//                         SUBCASE(testName_half.c_str()) {
//                             CHECK_NOTHROW(test_function.template operator()<sycl::half>(
//                                 q, width, batch_size, activation, output_activation, weight_init_mode));
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
