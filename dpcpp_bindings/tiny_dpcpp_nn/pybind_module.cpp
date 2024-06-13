/**
 * @file pybind_module.cpp
 * @author Kai Yuan
 * @brief
 * @version 0.1
 * @date 2024-01-22
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tnn_api.h" // need to include json before pybind11_json
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic conversion
#include <pybind11_json.hpp>
#include <torch/extension.h>

using json = nlohmann::json;
// C++ interface

class PybindingModule {
  public:
    PybindingModule(tnn::Module *module) : m_module{std::unique_ptr<tnn::Module>(module)} {}

    torch::Tensor fwd(torch::Tensor input) {
        CHECK_INPUT(input);
        return m_module->forward_pass(input);
    }

    torch::Tensor inference(torch::Tensor input) {
        CHECK_INPUT(input);
        return m_module->inference(input);
    }

    std::tuple<torch::Tensor, torch::Tensor> bwd_no_encoding_grad(torch::Tensor grad_output, bool pack_gradient,
                                                                  bool get_dl_dinput) {
        CHECK_INPUT(grad_output);
        return m_module->backward_pass(grad_output, pack_gradient, get_dl_dinput);
    }

    std::tuple<torch::Tensor, torch::Tensor> bwd_with_encoding_grad(torch::Tensor grad_output,
                                                                    torch::Tensor input_from_fwd, bool pack_gradient,
                                                                    bool get_dl_dinput) {
        CHECK_INPUT(grad_output);
        return m_module->backward_pass(grad_output, input_from_fwd, pack_gradient, get_dl_dinput);
    }

    torch::Tensor initial_params() { return m_module->initialize_params(); }

    torch::Tensor initial_params(torch::Tensor &tensor) { return m_module->initialize_params(tensor); }

    void set_params(torch::Tensor &tensor, bool weights_are_packed) {
        m_module->set_params(tensor, weights_are_packed);
    }

    torch::Tensor get_params() { return m_module->get_params(); }

    uint32_t n_params() const { return (uint32_t)m_module->n_params(); }
    uint32_t n_output_dims() const { return m_module->n_output_dims(); }

  private:
    std::unique_ptr<tnn::Module> m_module;
};

template <typename T, int WIDTH>
PybindingModule create_network_module(int input_width, int output_width, int n_hidden_layers, Activation activation,
                                      Activation output_activation) {
    tnn::NetworkModule<T, WIDTH> *network_module =
        new tnn::NetworkModule<T, WIDTH>(input_width, output_width, n_hidden_layers, activation, output_activation);
    return PybindingModule{network_module};
}

PybindingModule create_network_factory(int input_width, int output_width, int n_hidden_layers, Activation activation,
                                       Activation output_activation, int width, std::string dtype) {
    if (dtype == "torch.float16") {
        if (width == 16) {
            return create_network_module<sycl::half, 16>(input_width, output_width, n_hidden_layers, activation,
                                                         output_activation);
        } else if (width == 32) {
            return create_network_module<sycl::half, 32>(input_width, output_width, n_hidden_layers, activation,
                                                         output_activation);
        } else if (width == 64) {
            return create_network_module<sycl::half, 64>(input_width, output_width, n_hidden_layers, activation,
                                                         output_activation);
        } else if (width == 128) {
            return create_network_module<sycl::half, 128>(input_width, output_width, n_hidden_layers, activation,
                                                          output_activation);
        } else {
            throw std::runtime_error("Unsupported width.");
        }
    } else if (dtype == "torch.bfloat16") {
        if (width == 16) {
            return create_network_module<bf16, 16>(input_width, output_width, n_hidden_layers, activation,
                                                   output_activation);
        } else if (width == 32) {
            return create_network_module<bf16, 32>(input_width, output_width, n_hidden_layers, activation,
                                                   output_activation);
        } else if (width == 64) {
            return create_network_module<bf16, 64>(input_width, output_width, n_hidden_layers, activation,
                                                   output_activation);
        } else if (width == 128) {
            return create_network_module<bf16, 128>(input_width, output_width, n_hidden_layers, activation,
                                                    output_activation);
        } else {
            throw std::runtime_error("Unsupported width.");
        }
    } else {
        throw std::runtime_error("Unsupported dtype: " + dtype + ". Only fp16 and bf16 are supported");
    }
}
template <typename T> PybindingModule create_encoding_module(std::string encoding_name, const json &encoding_config) {
    tnn::EncodingModule<T> *encoding_module = new tnn::EncodingModule<T>(encoding_config);
    return PybindingModule{encoding_module};
}

template <typename T_enc, typename T_net, int WIDTH>
PybindingModule create_networkwithencoding_module(int input_width, int output_width, int n_hidden_layers,
                                                  Activation activation, Activation output_activation,
                                                  const json &encoding_config) {
    tnn::NetworkWithEncodingModule<T_enc, T_net, WIDTH> *networkwithencoding_module =
        new tnn::NetworkWithEncodingModule<T_enc, T_net, WIDTH>(input_width, output_width, n_hidden_layers, activation,
                                                                output_activation, encoding_config);
    return PybindingModule{networkwithencoding_module};
}

// C++ factory function
PybindingModule create_networkwithencoding_factory(int input_width, int output_width, int n_hidden_layers,
                                                   Activation activation, Activation output_activation,
                                                   const json &encoding_config, int width, std::string dtype) {

    if (dtype == "torch.float16") {
        if (width == 16) {
            return create_networkwithencoding_module<float, sycl::half, 16>(
                input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config);
        } else if (width == 32) {
            return create_networkwithencoding_module<float, sycl::half, 32>(
                input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config);
        } else if (width == 64) {
            return create_networkwithencoding_module<float, sycl::half, 64>(
                input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config);
        } else if (width == 128) {
            return create_networkwithencoding_module<float, sycl::half, 128>(
                input_width, output_width, n_hidden_layers, activation, output_activation, encoding_config);
        } else {
            throw std::runtime_error("Unsupported width.");
        }
    } else if (dtype == "torch.bfloat16") {
        if (width == 16) {
            return create_networkwithencoding_module<float, bf16, 16>(input_width, output_width, n_hidden_layers,
                                                                      activation, output_activation, encoding_config);
        } else if (width == 32) {
            return create_networkwithencoding_module<float, bf16, 32>(input_width, output_width, n_hidden_layers,
                                                                      activation, output_activation, encoding_config);
        } else if (width == 64) {
            return create_networkwithencoding_module<float, bf16, 64>(input_width, output_width, n_hidden_layers,
                                                                      activation, output_activation, encoding_config);
        } else if (width == 128) {
            return create_networkwithencoding_module<float, bf16, 128>(input_width, output_width, n_hidden_layers,
                                                                       activation, output_activation, encoding_config);
        } else {
            throw std::runtime_error("Unsupported width.");
        }
    } else {
        throw std::runtime_error("Unsupported dtype: " + dtype + ". Only fp16 and bf16 are supported");
    }
}

PYBIND11_MODULE(tiny_dpcpp_nn_pybind_module, m) {
    pybind11::enum_<Activation>(m, "Activation")
        .value("ReLU", Activation::ReLU)
        .value("LeakyReLU", Activation::LeakyReLU)
        .value("Exponential", Activation::Exponential)
        .value("Sine", Activation::Sine)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Squareplus", Activation::Squareplus)
        .value("Softplus", Activation::Softplus)
        .value("Tanh", Activation::Tanh)
        .value("Linear", Activation::None)
        .export_values();

    pybind11::class_<PybindingModule>(m, "Module")
        .def("fwd", &PybindingModule::fwd)
        .def("inference", &PybindingModule::inference)
        .def("bwd_no_encoding_grad", &PybindingModule::bwd_no_encoding_grad)
        .def("bwd_with_encoding_grad", &PybindingModule::bwd_with_encoding_grad)
        .def("initial_params", pybind11::overload_cast<>(&PybindingModule::initial_params))
        .def("initial_params", pybind11::overload_cast<torch::Tensor &>(&PybindingModule::initial_params))
        .def("set_params", &PybindingModule::set_params)
        .def("get_params", &PybindingModule::get_params)
        .def("n_output_dims", &PybindingModule::n_output_dims)
        .def("n_params", &PybindingModule::n_params);
    m.def("create_network", &create_network_factory, pybind11::arg("input_width"), pybind11::arg("output_width"),
          pybind11::arg("n_hidden_layers"), pybind11::arg("activation"), pybind11::arg("output_activation"),
          pybind11::arg("width"), pybind11::arg("dtype"));
    m.def("create_encoding", &create_encoding_module<float>); /// TODO: add a factory here later
    m.def("create_networkwithencoding", &create_networkwithencoding_factory, pybind11::arg("input_width"),
          pybind11::arg("output_width"), pybind11::arg("n_hidden_layers"), pybind11::arg("activation"),
          pybind11::arg("output_activation"), pybind11::arg("encoding_config"), pybind11::arg("width"),
          pybind11::arg("dtype"));
}
