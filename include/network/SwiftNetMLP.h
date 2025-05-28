/**
 * @file SwiftNetMLP.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a fused MLP class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <sycl/sycl.hpp>
#include <functional>
#include <iostream>
#include <json.hpp>
#include <unordered_map>
#include <vector>

#include "DeviceMatrix.h"
#include "Network.h"
#include "common.h"
// #include "kernel.h"
#include "kernel_esimd.h"

namespace syclex = sycl::ext::oneapi::experimental;

// Unique identifier for each activation pair
constexpr int ActivationPairCode(Activation act, Activation out_act) {
    return static_cast<int>(act) * 100 + static_cast<int>(out_act);
    // just has to be larger than the maximum amount of enumerations (3 with None,
    // Relu, and Sigmoid activation)
}

template <typename T, int WIDTH> class SwiftNetMLP : public Network<T> {
  public:
    /**
     * Constructor for the SwiftNetMLP class.
     *
     * @param q                  SYCL queue for command submission.
     * @param input_width        Width of the input data.
     * @param output_width       Width of the output data.
     * @param n_hidden_layers    Number of hidden layers.
     * @param activation         Activation function for hidden layers.
     * @param output_activation  Activation function for the output layer.
     * @tparam WIDTH             Width of the matrices.
     */
    SwiftNetMLP(sycl::queue &q, const int input_width, const int output_width, const int n_hidden_layers,
                Activation activation, Activation output_activation,
                const Network<T>::WeightInitMode mode = Network<T>::WeightInitMode::xavier_normal,
                const bool use_bias = false)
        : Network<T>(q, n_hidden_layers, input_width, WIDTH, output_width, mode, use_bias), m_activation{activation},
          m_output_activation{output_activation} {

        SanityCheck();
    }

    ~SwiftNetMLP() {}

    /**
     * Perform a forward pass of the SwiftNetMLP model.
     *
     * @param input The input data on the device.
     * @param forward Pointer to the forward intermediate array.
     * The output is stored at the end of the array 'forward'
     */
    std::vector<sycl::event> forward_pass(const DeviceMatrix<T> &input, DeviceMatrices<T> &intermediate_output_forward,
                                          const std::vector<sycl::event> &deps) override {
        SanityCheckForward(input, intermediate_output_forward);
        return forward_pass(input.GetView(), intermediate_output_forward.GetViews(), deps);
    }

    std::vector<sycl::event> forward_pass(const DeviceMatrixView<T> &input,
                                          DeviceMatricesView<T> intermediate_output_forward,
                                          const std::vector<sycl::event> &deps) override {

        using namespace tinydpcppnn::kernels::esimd;

#define ARGS_                                                                                                          \
    Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(), input, intermediate_output_forward,        \
        Network<T>::get_n_hidden_layers(), deps

        // Perform forward pass based on activation function
        switch (ActivationPairCode(m_activation, m_output_activation)) {
        case ActivationPairCode(Activation::None, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::None>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::None>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::None>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::None, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::Sigmoid>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::Sigmoid>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::Sigmoid>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::None, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::ReLU>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::ReLU>::forward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::ReLU>::forward_impl(ARGS_);
            break;
        default:
            throw std::logic_error("Invalid activation should have been caught earlier.");
        }

#undef ARGS_
    }

    /**
     * Perform a forward pass of the SwiftNetMLP model.
     *
     * @param input The input data on the device.
     * @param forward Pointer to the forward intermediate array. In inference this is not used for intermediate values.
     * The output is stored at the end of the array 'forward'
     */
    std::vector<sycl::event> inference(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                                       const std::vector<sycl::event> &deps) override {
        SanityCheckInference(input, output);

        return inference(input.GetView(), output.GetViews(), deps);
    }

    std::vector<sycl::event> inference(const DeviceMatrixView<T> &input, DeviceMatricesView<T> output,
                                       const std::vector<sycl::event> &deps) override {

        using namespace tinydpcppnn::kernels::esimd;

#define ARGS_                                                                                                          \
    Network<T>::get_queue(), Network<T>::get_weights_matrices().GetViews(), input, output,                             \
        Network<T>::get_n_hidden_layers(), deps

        // Perform forward pass based on activation function
        switch (ActivationPairCode(m_activation, m_output_activation)) {
        case ActivationPairCode(Activation::None, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::None>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::None>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::None>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::None, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::ReLU>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::ReLU>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::ReLU>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::None, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::Sigmoid>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::Sigmoid>::inference_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::Sigmoid>::inference_impl(
                ARGS_);
            break;
        default:
            throw std::logic_error("Invalid activation should have been caught earlier.");
        }

#undef ARGS_
    }

    /**
     * Perform the backward pass of the neural network.
     *
     * @param grads The gradients on the device. Input for the backward pass
     * @param out_inter Intermediate array for storing outputs. This is filled as part of the backward pass
     * @param forward Pointer to the forward intermediate array which was filled in the forw pass
     */
    std::vector<sycl::event> backward_pass(const DeviceMatrix<T> &input, DeviceMatrices<T> &output,
                                           DeviceMatrices<T> &intermediate_output_backward,
                                           const DeviceMatrices<T> &intermediate_output_forward,
                                           const std::vector<sycl::event> &deps,
                                           DeviceMatrixView<T> *dL_dinput = nullptr) override {
        SanityCheckBackward(input, output, intermediate_output_backward, intermediate_output_forward);

        return backward_pass(input.GetView(), output.GetViews(), intermediate_output_backward.GetViews(),
                             intermediate_output_forward.GetViews(), deps, dL_dinput);
    }

    std::vector<sycl::event> backward_pass(const DeviceMatrixView<T> input, DeviceMatricesView<T> output,
                                           DeviceMatricesView<T> intermediate_output_backward,
                                           const DeviceMatricesView<T> intermediate_output_forward,
                                           const std::vector<sycl::event> &deps,
                                           DeviceMatrixView<T> *dL_dinput = nullptr) override {
        std::optional<DeviceMatrixView<T>> dL_dinput_view;
        if (dL_dinput != nullptr) {
            dL_dinput_view.emplace(*dL_dinput);
        }
        using namespace tinydpcppnn::kernels::esimd;

#define ARGS_                                                                                                          \
    Network<T>::get_queue(), Network<T>::get_weightsT_matrices().GetViews(), input, output,                            \
        intermediate_output_backward, intermediate_output_forward, Network<T>::get_n_hidden_layers(), deps,            \
        dL_dinput_view

        switch (ActivationPairCode(m_activation, m_output_activation)) {
        case ActivationPairCode(Activation::None, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::None>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::None>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::None):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::None>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::None, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::ReLU>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::ReLU>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::ReLU):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::ReLU>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::None, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::None, Activation::Sigmoid>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::ReLU, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::ReLU, Activation::Sigmoid>::backward_impl(ARGS_);
            break;
        case ActivationPairCode(Activation::Sigmoid, Activation::Sigmoid):
            return EsimdKernels<T, WIDTH, WIDTH, WIDTH, Activation::Sigmoid, Activation::Sigmoid>::backward_impl(ARGS_);
            break;
        default:
            throw std::logic_error("Invalid activation should have been caught earlier.");
        }

#undef ARGS_
    }

  private:
    /// Generate the relevant kernel class. Has to be called in constructor
    void checkKernels() const {

	const auto arch = Network<T>::get_queue().get_device().template get_info<syclex::info::device::architecture>();
	switch (arch) {
	    case syclex::architecture::intel_gpu_pvc:
	    case syclex::architecture::intel_gpu_pvc_vg:
	    case syclex::architecture::intel_gpu_bmg_g21:
	    case syclex::architecture::intel_gpu_lnl_m:
		if (TARGET_DEVICE != 0) {
	            throw std::logic_error("Code built for PVC or Xe2 but tried to run on different device. (Set by cmake variable TARGET_DEVICE.)");
		}
		break;
	    case syclex::architecture::intel_gpu_acm_g10:
	    case syclex::architecture::intel_gpu_acm_g11:
	    case syclex::architecture::intel_gpu_acm_g12:
		if (TARGET_DEVICE != 1) {
	            throw std::logic_error("Code built for Alchemist but tried to run on different device. (Set by cmake variable TARGET_DEVICE.)");
		}
		break;
	    default:
	    	throw std::logic_error("Unsupported GPU.");
	}
    }

    // TODO: does this have to be virtual?
    virtual void SanityCheck() const override {
        static_assert(WIDTH == 16 || WIDTH == 32 || WIDTH == 64 || WIDTH == 128);
        static_assert(std::is_same<T, sycl::ext::oneapi::bfloat16>::value || std::is_same<T, sycl::half>::value);

        if (m_activation != Activation::ReLU && m_activation != Activation::None &&
            m_activation != Activation::Sigmoid) {
            throw std::runtime_error("m_activation must be ReLU or None or Sigmoid for now.");
        }
        if (m_output_activation != Activation::ReLU && m_output_activation != Activation::None &&
            m_output_activation != Activation::Sigmoid) {
            throw std::runtime_error("m_output_activation must be ReLU or None or Sigmoid for now.");
        }

        checkKernels();
    }

    void SanityCheckInference(const DeviceMatrix<T> &input, DeviceMatrix<T> &output) const {
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

        if (input.n() != Network<T>::get_input_width()) {
            throw std::invalid_argument("Input array too small. Input n: " + std::to_string(input.n()) +
                                        " != input width " + std::to_string(Network<T>::get_input_width()));
        }
        if (output.m() != input.m() || output.n() < Network<T>::get_output_width())
            throw std::invalid_argument("Output array too small");
    }

    void SanityCheckForward(const DeviceMatrix<T> &input, DeviceMatrices<T> &intermediate_output_forward) const {
        // Static assertion and assertion checks
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

        if (input.n() != Network<T>::get_input_width()) {
            throw std::invalid_argument("Input array too small. Input n: " + std::to_string(input.n()) +
                                        " != input width " + std::to_string(Network<T>::get_input_width()));
        }

        if (intermediate_output_forward.input_m() != input.m() || intermediate_output_forward.input_n() != input.n()) {
            throw std::invalid_argument("intermediate_output_forward dim wrong doesn't match input shape, expected: (" +
                                        std::to_string(input.m()) + ", " + std::to_string(input.n()) + "), but got: (" +
                                        std::to_string(intermediate_output_forward.input_m()) + ", " +
                                        std::to_string(intermediate_output_forward.input_n()) + ")");
        }

        if (intermediate_output_forward.middle_m() != input.m() || intermediate_output_forward.middle_n() != WIDTH) {
            throw std::invalid_argument("intermediate_output_forward array too small for middle results, expected: (" +
                                        std::to_string(input.m()) + ", " + std::to_string(WIDTH) + "), but got: (" +
                                        std::to_string(intermediate_output_forward.middle_m()) + ", " +
                                        std::to_string(intermediate_output_forward.middle_n()) + ")");
        }

        if (intermediate_output_forward.output_m() != input.m() ||
            intermediate_output_forward.output_n() != Network<T>::get_output_width()) {
            throw std::invalid_argument("intermediate_output_forward array too small for output, expected: (" +
                                        std::to_string(input.m()) + ", " +
                                        std::to_string(Network<T>::get_output_width()) + "), but got: (" +
                                        std::to_string(intermediate_output_forward.output_m()) + ", " +
                                        std::to_string(intermediate_output_forward.output_n()) + ")");
        }

        if (int(intermediate_output_forward.GetNumberOfMatrices()) != Network<T>::get_n_hidden_layers() + 2) {
            throw std::invalid_argument("Not enough matrices in intermediate_output_forward array, expected: " +
                                        std::to_string(Network<T>::get_n_hidden_layers() + 2) + " but got: " +
                                        std::to_string(intermediate_output_forward.GetNumberOfMatrices()));
        }
    }

    void SanityCheckBackward(const DeviceMatrix<T> &input, DeviceMatrices<T> &output,
                             DeviceMatrices<T> &intermediate_output_backward,
                             const DeviceMatrices<T> &intermediate_output_forward) const {
        if ((input.m() % 8) != 0) throw std::invalid_argument("Batch size is not divisible by 8.");

        if (input.m() != intermediate_output_forward.input_m() || input.n() != Network<T>::get_output_width())
            throw std::invalid_argument("Input array in backward pass too small. Input shape: " +
                                        std::to_string(input.m()) + " x " + std::to_string(input.n()) +
                                        ", but should be: " + std::to_string(intermediate_output_forward.input_m()) +
                                        " x " + std::to_string(Network<T>::get_output_width()));

        if (intermediate_output_forward.input_m() != input.m() || intermediate_output_forward.middle_m() != input.m() ||
            intermediate_output_forward.output_m() != input.m())
            throw std::invalid_argument(
                "Intermediate_output_forward m dimension mismatch. Expected: " + std::to_string(input.m()) +
                " for all sizes, but got input_m=" + std::to_string(intermediate_output_forward.input_m()) +
                ", middle_m=" + std::to_string(intermediate_output_forward.middle_m()) +
                ", output_m=" + std::to_string(intermediate_output_forward.output_m()));

        if (intermediate_output_forward.input_n() != Network<T>::get_input_width() ||
            intermediate_output_forward.middle_n() != WIDTH ||
            intermediate_output_forward.output_n() != Network<T>::get_output_width())
            throw std::invalid_argument("Intermediate_output_forward n dimension mismatch. Expected: (" +
                                        std::to_string(Network<T>::get_input_width()) + ", " + std::to_string(WIDTH) +
                                        ", " + std::to_string(Network<T>::get_output_width()) +
                                        ") for (input_n, middle_n, output_n), " + "but got (" +
                                        std::to_string(intermediate_output_forward.input_n()) + ", " +
                                        std::to_string(intermediate_output_forward.middle_n()) + ", " +
                                        std::to_string(intermediate_output_forward.output_n()) + ")");

        if (int(intermediate_output_forward.GetNumberOfMatrices()) != Network<T>::get_n_hidden_layers() + 2)
            throw std::invalid_argument("Not enough matrices in intermediate_output_forward array. Required: " +
                                        std::to_string(Network<T>::get_n_hidden_layers() + 2) + ", available: " +
                                        std::to_string(intermediate_output_forward.GetNumberOfMatrices()));

        if (intermediate_output_backward.input_m() != input.m() ||
            intermediate_output_backward.middle_m() != input.m() ||
            intermediate_output_backward.output_m() != input.m())
            throw std::invalid_argument(
                "Intermediate_output_backward m too small. Expected: " + std::to_string(input.m()) +
                " for all m sizes, but got input_m=" + std::to_string(intermediate_output_backward.input_m()) +
                ", middle_m=" + std::to_string(intermediate_output_backward.middle_m()) +
                ", output_m=" + std::to_string(intermediate_output_backward.output_m()));

        if (intermediate_output_backward.input_n() != WIDTH || intermediate_output_backward.middle_n() != WIDTH ||
            intermediate_output_backward.output_n() != Network<T>::get_output_width())
            throw std::invalid_argument("Intermediate_output_backward n dimension mismatch. Expected: (" +
                                        std::to_string(WIDTH) + ", " + std::to_string(WIDTH) + ", " +
                                        std::to_string(Network<T>::get_output_width()) +
                                        ") for (input_n, middle_n, output_n), but got (" +
                                        std::to_string(intermediate_output_backward.input_n()) + ", " +
                                        std::to_string(intermediate_output_backward.middle_n()) + ", " +
                                        std::to_string(intermediate_output_backward.output_n()) + ")");

        if (int(intermediate_output_backward.GetNumberOfMatrices()) != Network<T>::get_n_hidden_layers() + 1)
            throw std::invalid_argument("Not enough matrices in intermediate_output_backward array. Required: " +
                                        std::to_string(Network<T>::get_n_hidden_layers() + 1) + ", available: " +
                                        std::to_string(intermediate_output_backward.GetNumberOfMatrices()));

        if (output.input_m() != Network<T>::get_input_width() || output.input_n() != WIDTH)
            throw std::invalid_argument("Output of backward pass too small for input. Expected: (" +
                                        std::to_string(Network<T>::get_input_width()) + ", " + std::to_string(WIDTH) +
                                        "), but got (" + std::to_string(output.input_m()) + ", " +
                                        std::to_string(output.input_n()) + ")");

        if (output.middle_m() != WIDTH || output.middle_n() != WIDTH)
            throw std::invalid_argument("Output of backward pass too small for middle. Expected: (" +
                                        std::to_string(WIDTH) + ", " + std::to_string(WIDTH) + "), but got (" +
                                        std::to_string(output.middle_m()) + ", " + std::to_string(output.middle_n()) +
                                        ")");

        if (output.output_m() != WIDTH || output.output_n() != Network<T>::get_output_width())
            throw std::invalid_argument("Output of backward pass too small for output. Expected: (" +
                                        std::to_string(WIDTH) + ", " + std::to_string(Network<T>::get_output_width()) +
                                        "), but got (" + std::to_string(output.output_m()) + ", " +
                                        std::to_string(output.output_n()) + ")");
    }

    Activation m_activation;
    Activation m_output_activation;
};

template <typename T, int WIDTH>
std::shared_ptr<SwiftNetMLP<T, WIDTH>> create_network(sycl::queue &q, const int output_width, const int n_hidden_layers,
                                                      Activation activation, Activation output_activation) {
    return std::make_shared<SwiftNetMLP<T, WIDTH>>(q, WIDTH, output_width, n_hidden_layers, activation,
                                                   output_activation);
}
