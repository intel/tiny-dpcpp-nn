/**
 * @file trainer.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a class which represents a network trainer.
 * TODO: actually implement this with optimizers and weights updates.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "tnn_api.h"

template <typename T> class Trainer {
  public:
    Trainer(tnn::Module *network, float weight_val) : m_network(network) {
        torch::Tensor init_params =
            torch::ones({(int)m_network->n_params(), 1}).to(torch::kXPU).to(c10::ScalarType::BFloat16) * weight_val;
        m_params = m_network->initialize_params(init_params);
    }

    torch::Tensor training_step(torch::Tensor &input, torch::Tensor &dL_doutput) {
        const float input_val = 1.0;

        auto output_net = m_network->forward_pass(input);
        m_network->backward_pass(dL_doutput, false, false);
        return output_net;
    }

  private:
    tnn::Module *m_network;
    torch::Tensor m_params;
};
