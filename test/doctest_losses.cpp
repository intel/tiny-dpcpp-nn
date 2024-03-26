/**
 * @file doctest_losses.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the losses.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// This file includes tests for all of our loss calculations

#include "doctest/doctest.h"

#include "l1.h"
#include "l2.h"

template <typename T> std::vector<T> create_vector(size_t size, T value) { return std::vector<T>(size, value); }

void test_l2_loss_with_parameters(sycl::queue &q, size_t output_dim, size_t batch_size, float loss_scale,
                                  float prediction_value, float target_value) {

    const size_t n_elements = batch_size * output_dim;

    // Initialize matrices with test values
    DeviceMatrix<float> predictions(batch_size, output_dim, q);
    DeviceMatrix<float> targets(batch_size, output_dim, q);
    DeviceMatrix<float> values(batch_size, output_dim, q);
    DeviceMatrix<float> gradients(batch_size, output_dim, q);
    predictions.fill(prediction_value).wait();
    targets.fill(target_value).wait();
    values.fill(0.0).wait();
    gradients.fill(0.0).wait();

    // Create L2Loss instance and run evaluate
    L2Loss<float> loss;
    sycl::event loss_event = loss.evaluate(q, loss_scale, predictions.GetView(), targets, values, gradients);
    loss_event.wait();

    // Verify the results are as expected
    auto host_values = values.copy_to_host();
    auto host_gradients = gradients.copy_to_host();

    float expected_value = (prediction_value - target_value) * (prediction_value - target_value) / n_elements;
    float expected_gradient = loss_scale * 2.0f * (prediction_value - target_value) / n_elements;
    for (size_t i = 0; i < n_elements; ++i) {
        CHECK(host_values[i] == doctest::Approx(expected_value));
        CHECK(host_gradients[i] == doctest::Approx(expected_gradient));
    }
}
void test_l1_loss_with_parameters(sycl::queue &q, size_t output_dim, size_t batch_size, float loss_scale,
                                  float prediction_value, float target_value) {

    const size_t n_elements = batch_size * output_dim;

    // Initialize matrices with test values
    DeviceMatrix<float> predictions(batch_size, output_dim, q);
    DeviceMatrix<float> targets(batch_size, output_dim, q);
    DeviceMatrix<float> values(batch_size, output_dim, q);
    DeviceMatrix<float> gradients(batch_size, output_dim, q);
    predictions.fill(prediction_value).wait();
    targets.fill(target_value).wait();
    values.fill(0.0).wait();
    gradients.fill(0.0).wait();

    // Create L1Loss instance and run evaluate
    L1Loss<float> loss;
    sycl::event loss_event = loss.evaluate(q, loss_scale, predictions.GetView(), targets, values, gradients);
    loss_event.wait();

    // Verify the results are as expected
    auto host_values = values.copy_to_host();
    auto host_gradients = gradients.copy_to_host();

    float expected_value = std::abs(prediction_value - target_value) / n_elements;
    float expected_gradient =
        loss_scale * (prediction_value > target_value ? 1.0f : (prediction_value < target_value ? -1.0f : 0.0f)) /
        n_elements;

    for (size_t i = 0; i < n_elements; ++i) {
        CHECK(host_values[i] == doctest::Approx(expected_value));
        CHECK(host_gradients[i] == doctest::Approx(expected_gradient));
    }
}
TEST_CASE("L2Loss Computes Correctly for Multiple Dimensions and Prediction Values") {
    sycl::queue q;

    SUBCASE("Batch size 8, dimension 16, positive predictions") {
        test_l2_loss_with_parameters(q, 16, 8, 1.0f, 2.56f, 0.1f);
    } // this is the loss for swiftnet bwd unittest

    SUBCASE("Batch size 1, dimension 100, positive predictions") {
        test_l2_loss_with_parameters(q, 100, 1, 101.0f, 2.0f, 1.0f);
    }

    SUBCASE("Batch size 1, dimension 300, positive predictions") {
        test_l2_loss_with_parameters(q, 300, 1, 101.0f, 2.0f, 1.0f);
    }

    SUBCASE("Batch size 2, dimension 100, negative predictions") {
        test_l2_loss_with_parameters(q, 100, 2, 101.0f, -2.0f, -1.0f);
    }

    SUBCASE("Batch size 3, dimension 50, mixed predictions") {
        test_l2_loss_with_parameters(q, 50, 3, 101.0f, -1.5f, 2.5f);
    }
}
TEST_CASE("L1Loss Computes Correctly for Multiple Dimensions and Prediction Values") {
    sycl::queue q;
    SUBCASE("Batch size 8, dimension 16, positive predictions") {
        test_l1_loss_with_parameters(q, 16, 8, 1.0f, 2.56f, 0.1f);
    } // this is the loss for swiftnet bwd unittest

    SUBCASE("Batch size 1, dimension 100, positive predictions") {
        test_l1_loss_with_parameters(q, 100, 1, 111.0f, 2.0f, 1.0f);
    }

    SUBCASE("Batch size 1, dimension 300, positive predictions") {
        test_l1_loss_with_parameters(q, 300, 1, 123.0f, 3.0f, 2.0f);
    }

    SUBCASE("Batch size 2, dimension 100, negative predictions") {
        test_l1_loss_with_parameters(q, 100, 2, 234.0f, -2.0f, -1.0f);
    }

    SUBCASE("Batch size 3, dimension 50, mixed predictions") {
        test_l1_loss_with_parameters(q, 50, 3, 323.0f, -1.5f, 0.5f);
    }
}