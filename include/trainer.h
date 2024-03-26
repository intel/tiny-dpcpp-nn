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

#include "DeviceMatrix.h"
#include "Network.h"

template <typename T> class Trainer {
  public:
    Trainer(Network<T> *network) : m_network(network) {}

    std::vector<sycl::event> training_step(const DeviceMatrix<T> &input, DeviceMatrices<T> &output,
                                           DeviceMatrix<T> &losses, DeviceMatrices<T> &out_inter_forw,
                                           DeviceMatrices<T> &out_inter_backw,
                                           const std::vector<sycl::event> &dependencies) {

        // return m_network->training(input, target, m_network->m_forward, m_network->m_out_inter, deps);
        auto deps = m_network->forward_pass(input, out_inter_forw, dependencies);
        // auto e = m_loss->evaluate(m_network->get_queue(), scale, output, target, grads, losses);
        // deps = {e};

        deps = m_network->backward_pass(losses, output, out_inter_backw, out_inter_forw, deps);

        // no optimisation as we run benchmarking
        // m_optim->step(m_network->get_queue(), scale,
        // m_network->m_weights_matrices,
        //               m_network->m_weightsT_matrices,
        //               m_network->m_grads_matrices, WIDTH);

        return deps;
    }

  private:
    Network<T> *m_network;
};
