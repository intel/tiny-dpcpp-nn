/**
 * @file adam.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of an Adam optimizer class.
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
class AdamOptimizer : public Optimizer {
  public:
    void step(queue q, float loss_scale, DeviceMem<bf16> &weights, DeviceMem<bf16> &weightsT,
              DeviceMem<bf16> &gradients, int WIDTH) override;

    void set_learning_rate(const float learning_rate);

  private:
    DeviceMem<float> m_first_moments;
    DeviceMem<float> m_second_moments;

    // int m_output_rows;
    // int m_n_hidden_layers;
    float m_learning_rate = 1e-3f;
    float m_l2_reg = 1e-8f;
};
