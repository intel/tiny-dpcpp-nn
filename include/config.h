/**
 * @file config.h
 * @author Kai Yuan
 * @brief Implementation of I do not even know what.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "L1.h"
#include "L2.h"
#include "Network.h"
#include "RelativeL1.h"
#include "RelativeL2.h"
#include "SwiftNetMLP.h"
#include "adam.h"
#include "cross_entropy.h"
#include "json.hpp"
#include "loss.h"
#include "optimizer.h"
#include "sgd.h"
#include "trainer.h"

using json = nlohmann::json;

Activation string_to_activation(const std::string &activation_name) {
    if (isequalstring(activation_name, "None")) {
        return Activation::None;
    } else if (isequalstring(activation_name, "ReLU")) {
        return Activation::ReLU;
    } else if (isequalstring(activation_name, "Exponential")) {
        return Activation::Exponential;
    } else if (isequalstring(activation_name, "Sigmoid")) {
        return Activation::Sigmoid;
    } else if (isequalstring(activation_name, "Sine")) {
        return Activation::Sine;
    } else if (isequalstring(activation_name, "Tanh")) {
        return Activation::Tanh;
    }
    throw std::runtime_error{"Invalid activation name:}"};
}

struct TrainableModel {
    queue m_q;
    Loss *loss;
    Optimizer *optimizer;
    Network *network;
    Trainer trainer;
};

TrainableModel create_from_config(queue q, json config) {
    queue m_q{q};
    std::string loss_type = config.value("loss", json::object()).value("otype", "RelativeL2");
    std::string optimizer_type = config.value("optimizer", json::object()).value("otype", "sgd");
    const int WIDTH = config.value("network", json::object()).value("n_neurons", 64);

    Loss *loss;
    Optimizer *optimizer;
    Network *network;

    if (isequalstring(loss_type, "L2")) {
        loss = new L2Loss();
    } else if (isequalstring(loss_type, "RelativeL2")) {
        loss = new RelativeL2Loss();
    } else if (isequalstring(loss_type, "L1")) {
        loss = new L1Loss();
    } else if (isequalstring(loss_type, "RelativeL1")) {
        loss = new RelativeL1Loss();
    } else if (isequalstring(loss_type, "CrossEntropy")) {
        loss = new CrossEntropyLoss();
    } else {
        throw std::runtime_error{"Invalid loss type: "};
    }

    if (isequalstring(optimizer_type, "Adam")) {
        optimizer = new AdamOptimizer();
    }

    else if (isequalstring(optimizer_type, "SGD")) {
        optimizer = new SGDOptimizer(config.value("optimizer", json::object()).value("output_width", 64),
                                     config.value("optimizer", json::object()).value("n_hidden_layer", 2),
                                     config.value("optimizer", json::object()).value("learning_rate", 1e-3f),
                                     config.value("optimizer", json::object()).value("l2_reg", 1e-8f));
    } else {
        throw std::runtime_error{"Invalid optimizer type: "};
    }

    switch (WIDTH) {
    case 64:
        network = new SwiftNetMLP<64>(
            q, config.value("network", json::object()).value("n_input_dims", 64),
            config.value("network", json::object()).value("n_output_dims", 64),
            config.value("network", json::object()).value("n_hidden_layers", 2),
            string_to_activation(config.value("network", json::object()).value("activation", "ReLU")),
            string_to_activation(config.value("network", json::object()).value("output_activation", "None")),
            config.value("network", json::object()).value("batch_size", 8192));
        break;
    case 128:
        network = new SwiftNetMLP<128>(
            q, config.value("network", json::object()).value("n_input_dims", 128),
            config.value("network", json::object()).value("n_output_dims", 128),
            config.value("network", json::object()).value("n_hidden_layers", 2),
            string_to_activation(config.value("network", json::object()).value("activation", "ReLU")),
            string_to_activation(config.value("network", json::object()).value("output_activation", "None")),
            config.value("network", json::object()).value("batch_size", 8192));
        break;
    default:
        throw std::runtime_error{"SwiftNetMLP only supports 64, and 128 neurons, but got ..."};
    }
    auto trainer = Trainer(*network, *loss, *optimizer);
    return {m_q, loss, optimizer, network, trainer};
}
