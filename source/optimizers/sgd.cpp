/**
 * @file sgd.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Definition of sgd optimizer class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "sgd.h"

/**
 * Perform a single step of Stochastic Gradient Descent (SGD) optimization for
 * updating weights.
 *
 * This function updates the weights of a neural network using the SGD
 * optimization algorithm. It computes the new weight values based on the
 * provided gradients, learning rate, L2 regularization factor, and loss scale.
 *
 * @param idx            The global index of the weight element to update.
 * @param n_elements     The total number of weight elements.
 * @param output_width   The width of the output layer.
 * @param n_hidden_layers The number of hidden layers in the network.
 * @param loss_scale     The scale factor for loss.
 * @param learning_rate  The learning rate for the optimization step.
 * @param l2_reg         The L2 regularization factor.
 * @param weights        Pointer to the array of weights.
 * @param gradients      Pointer to the array of gradients.
 * @param WIDTH          The width of weight matrices.
 */
void sgd_step(id<1> idx, const int n_elements, int output_width, int n_hidden_layers, const float loss_scale,
              const float learning_rate, const float l2_reg, bf16 *weights, bf16 *gradients, int WIDTH) {
    // Calculate the index of the current matrix and its offset within the matrix
    int matrices_number = idx / (WIDTH * WIDTH);
    int matrices_offset = idx % (WIDTH * WIDTH);
    int packed_idx_matrices = 0;

    // Determine if the matrix belongs to hidden layers or the output layer
    if (matrices_number < n_hidden_layers) {
        packed_idx_matrices = toPackedLayoutCoord(matrices_offset, WIDTH, WIDTH);
    } else {
        packed_idx_matrices = toPackedLayoutCoord(matrices_offset, WIDTH, output_width);
    }

    // Calculate the packed index for the current weight in the matrix
    const int packed_idx = matrices_number * WIDTH * WIDTH + packed_idx_matrices;
    const bf16 weight = weights[packed_idx];
    float gradient = gradients[idx];

    // Apply L2 regularization
    gradient += l2_reg * weight;

    // Calculate the new weight using the SGD update rule
    const bf16 new_weight = weight - learning_rate * gradient;

    // Update the weight
    weights[packed_idx] = new_weight;
}

// Perform a single step of SGD optimization for transposed weight matrices
/**
 * Perform a single step of Stochastic Gradient Descent (SGD) optimization for
 * updating transposed weights.
 *
 * This function updates the transposed weights of a neural network using the
 * SGD optimization algorithm. It computes the new transposed weight values
 * based on the provided gradients, learning rate, L2 regularization factor, and
 * loss scale.
 *
 * @param idx             The global index of the transposed weight element to
 * update.
 * @param n_elements      The total number of transposed weight elements.
 * @param output_width    The width of the output layer.
 * @param n_hidden_layers The number of hidden layers in the network.
 * @param loss_scale      The scale factor for loss.
 * @param learning_rate   The learning rate for the optimization step.
 * @param l2_reg          The L2 regularization factor.
 * @param weightsT        Pointer to the array of transposed weights.
 * @param gradients       Pointer to the array of gradients.
 * @param WIDTH           The width of weight matrices.
 */
void sgd_stepT(id<1> idx, const int n_elements, int output_width, int n_hidden_layers, const float loss_scale,
               const float learning_rate, const float l2_reg, bf16 *weightsT, bf16 *gradients, int WIDTH) {
    // Calculate the row and column indices of the transposed weight matrix
    const int i = idx / WIDTH;
    const int j = idx % WIDTH;

    // Calculate the transposed index
    const int T_idx = WIDTH * j + i;

    // Calculate the matrix number and offset within the matrix for transposed
    // weights
    const int matrices_number = T_idx / (WIDTH * WIDTH);
    const int matrices_offset = T_idx % (WIDTH * WIDTH);
    int packed_idx_matrices = 0;

    // Determine if the matrix belongs to hidden layers or the output layer
    if (matrices_number < n_hidden_layers) {
        packed_idx_matrices = fromPackedLayoutCoord(matrices_offset, WIDTH, WIDTH);
    } else {
        packed_idx_matrices = fromPackedLayoutCoord(matrices_offset, output_width, WIDTH);
    }

    // Calculate the packed index for the current transposed weight
    const int packed_idx = matrices_number * WIDTH * WIDTH + packed_idx_matrices;
    const bf16 weightT = weightsT[packed_idx];
    float gradient = gradients[idx] / loss_scale;

    // Apply L2 regularization
    gradient += l2_reg * weightT;

    // Calculate the new transposed weight using the SGD update rule
    const bf16 new_weightT = weightT - learning_rate * gradient;

    // Update the transposed weight
    weightsT[packed_idx] = new_weightT;
}

// Constructor for SGDOptimizer class
SGDOptimizer::SGDOptimizer(int output_rows, int n_hidden_layers, float learning_rate, float l2_reg) {
    // Initialize optimizer parameters
    m_output_rows = output_rows;
    m_n_hidden_layers = n_hidden_layers;
    m_learning_rate = learning_rate;
    m_l2_reg = l2_reg;
}

// Perform a step of SGD optimization using provided queue and loss scale
void SGDOptimizer::step(queue q, float loss_scale, DeviceMem<bf16> &weights, DeviceMem<bf16> &weightsT,
                        DeviceMem<bf16> &gradients, int WIDTH) {
    const int n_elements = weights.size();
    float learning_rate = m_learning_rate;
    float l2_reg = m_l2_reg;
    const int output_rows = m_output_rows;
    const int n_hidden_layers = m_n_hidden_layers;

    // Perform the SGD update for weight matrices
    q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
         sgd_step(idx, n_elements, output_rows, n_hidden_layers, loss_scale, learning_rate, l2_reg, weights.data(),
                  gradients.data(), WIDTH);
     }).wait();

    // Perform the SGD update for transposed weight matrices
    q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
         sgd_stepT(idx, n_elements, output_rows, n_hidden_layers, loss_scale, learning_rate, l2_reg, weightsT.data(),
                   gradients.data(), WIDTH);
     }).wait();
}

// Set the learning rate for the optimizer
void SGDOptimizer::set_learning_rate(const float learning_rate) { m_learning_rate = learning_rate; }
