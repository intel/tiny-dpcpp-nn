/**
 * @file sgd.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of sgd optimizer class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <vector>

#include "common.h"
#include "optimizer.h"
// #include "L2.h"
class SGDOptimizer : public Optimizer {
  public:
    SGDOptimizer(int output_rows, int n_hidden_layers, float learning_rate, float l2_reg);

    void step(queue q, float loss_scale, DeviceMem<bf16> &weights, DeviceMem<bf16> &weightsT,
              DeviceMem<bf16> &gradients, int WIDTH) override;

    void set_learning_rate(const float learning_rate);

  private:
    int m_output_rows;
    int m_n_hidden_layers;
    float m_learning_rate = 1e-3f;
    float m_l2_reg = 1e-8f;
};
