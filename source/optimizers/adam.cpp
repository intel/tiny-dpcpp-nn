/**
 * @file adam.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Definition of Adam optimizer class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "adam.h"

#include <vector>

/**
 * Perform an Adam optimizer step for a single element.
 *
 * @param idx Index of the element to process.
 * @param n_elements Total number of elements.
 * @param relative_weight_decay Relative weight decay coefficient.
 * @param absolute_weight_decay Absolute weight decay coefficient.
 * @param weight_clipping_magnitude Weight clipping magnitude.
 * @param loss_scale Loss scale factor.
 * @param learning_rate Learning rate.
 * @param non_matrix_learning_rate_factor Non-matrix learning rate factor.
 * @param beta1 Beta1 coefficient for first moment.
 * @param beta2 Beta2 coefficient for second moment.
 * @param epsilon Small value to prevent division by zero.
 * @param lower_lr_bound Lower bound for the learning rate.
 * @param upper_lr_bound Upper bound for the learning rate.
 * @param l2_reg L2 regularization coefficient.
 * @param weights Pointer to weights (bf16 type).
 * @param gradients Pointer to gradients (bf16 type).
 * @param first_moments Pointer to first moments.
 * @param second_moments Pointer to second moments.
 * @param WIDTH Width of the matrix (for matrix operations).
 */
void adam_step(id<1> idx, const int n_elements, const float relative_weight_decay, const float absolute_weight_decay,
               const float weight_clipping_magnitude, const float loss_scale, float learning_rate,
               const float non_matrix_learning_rate_factor, const float beta1, const float beta2, const float epsilon,
               const float lower_lr_bound, const float upper_lr_bound, const float l2_reg, bf16 *weights,
               const bf16 *gradients, float *first_moments, float *second_moments, int WIDTH) {
    const bf16 weight = weights[idx];
    bf16 gradient = gradients[idx] / loss_scale;

    gradient += l2_reg * weight;

    const float gradient_sq = gradient * gradient;

    float first_moment = first_moments[idx] = beta1 * first_moments[idx] + (1 - beta1) * gradient;
    const float second_moment = second_moments[idx] = beta2 * second_moments[idx] + (1 - beta2) * gradient_sq;

    const float effective_learning_rate =
        fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

    float new_weight = effective_learning_rate * first_moment;

    if (weight_clipping_magnitude != 0.0f) {
        new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
    }

    weights[idx] = (bf16)new_weight;
}

/**
 * Perform an Adam optimizer step for a single element for the transposed weight
 * matrix.
 *
 * @param idx Index of the element to process.
 * @param n_elements Total number of elements.
 * @param relative_weight_decay Relative weight decay coefficient.
 * @param absolute_weight_decay Absolute weight decay coefficient.
 * @param weight_clipping_magnitude Weight clipping magnitude.
 * @param loss_scale Loss scale factor.
 * @param learning_rate Learning rate.
 * @param non_matrix_learning_rate_factor Non-matrix learning rate factor.
 * @param beta1 Beta1 coefficient for first moment.
 * @param beta2 Beta2 coefficient for second moment.
 * @param epsilon Small value to prevent division by zero.
 * @param lower_lr_bound Lower bound for the learning rate.
 * @param upper_lr_bound Upper bound for the learning rate.
 * @param l2_reg L2 regularization coefficient.
 * @param weights Pointer to weights (bf16 type).
 * @param gradients Pointer to gradients (bf16 type).
 * @param first_moments Pointer to first moments.
 * @param second_moments Pointer to second moments.
 * @param WIDTH Width of the matrix (for matrix operations).
 */
void adam_stepT(id<1> idx, const int n_elements, const float relative_weight_decay, const float absolute_weight_decay,
                const float weight_clipping_magnitude, const float loss_scale, float learning_rate,
                const float non_matrix_learning_rate_factor, const float beta1, const float beta2, const float epsilon,
                const float lower_lr_bound, const float upper_lr_bound, const float l2_reg, bf16 *weightsT,
                const bf16 *gradients, float *first_moments, float *second_moments, int WIDTH) {
    const int i = idx / WIDTH;
    const int j = idx % WIDTH;

    const int T_idx = WIDTH * j + i;

    const bf16 weight = weightsT[T_idx];
    bf16 gradient = gradients[T_idx] / loss_scale;

    gradient += l2_reg * weight;

    const float gradient_sq = gradient * gradient;

    float first_moment = first_moments[T_idx] = beta1 * first_moments[T_idx] + (1 - beta1) * gradient;
    const float second_moment = second_moments[T_idx] = beta2 * second_moments[T_idx] + (1 - beta2) * gradient_sq;

    const float effective_learning_rate =
        fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

    float new_weight = effective_learning_rate * first_moment;

    if (weight_clipping_magnitude != 0.0f) {
        new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
    }

    weightsT[T_idx] = (bf16)new_weight;
}

/**
 * Perform Adam optimizer steps on a batch of elements.
 *
 * @param q SYCL queue for parallel computation.
 * @param loss_scale Loss scale factor.
 * @param weights Weights tensor (DeviceMem<bf16>).
 * @param weightsT Transposed weights tensor (DeviceMem<bf16>).
 * @param gradients Gradients tensor (DeviceMem<bf16>).
 * @param WIDTH Width of the matrix (for matrix operations).
 */
void AdamOptimizer::step(queue q, float loss_scale, DeviceMem<bf16> &weights, DeviceMem<bf16> &weightsT,
                         DeviceMem<bf16> &gradients, int WIDTH) {
    const int n_elements = weights.size();
    float learning_rate = m_learning_rate;
    float l2_reg = m_l2_reg;
    const float relative_weight_decay = 0.01f;
    const float absolute_weight_decay = 0.01f;
    const float weight_clipping_magnitude = 0.01f;
    const float non_matrix_learning_rate_factor = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.99f;
    const float epsilon = 0.01f;
    const float lower_lr_bound = 0.0001f;
    const float upper_lr_bound = 0.1f;

    auto first_moment = m_first_moments.data();
    auto second_moment = m_second_moments.data();

    q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
         adam_step(idx, n_elements, relative_weight_decay, absolute_weight_decay, weight_clipping_magnitude, loss_scale,
                   learning_rate, non_matrix_learning_rate_factor, beta1, beta2, epsilon, lower_lr_bound,
                   upper_lr_bound, l2_reg, weights.data(), gradients.data(), first_moment, second_moment, WIDTH);
     }).wait();

    q.parallel_for<>(range<1>(n_elements), [=](id<1> idx) {
         adam_stepT(idx, n_elements, relative_weight_decay, absolute_weight_decay, weight_clipping_magnitude,
                    loss_scale, learning_rate, non_matrix_learning_rate_factor, beta1, beta2, epsilon, lower_lr_bound,
                    upper_lr_bound, l2_reg, weightsT.data(), gradients.data(), first_moment, second_moment, WIDTH);
     }).wait();
}

/**
 * Set the learning rate for the Adam optimizer.
 *
 * @param learning_rate The new learning rate.
 */
void AdamOptimizer::set_learning_rate(const float learning_rate) { m_learning_rate = learning_rate; }
