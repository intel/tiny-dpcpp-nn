/**
 * @file Network.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of an abstract network class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "DeviceMatrix.h"
#include "common.h"
#include <random>

// completely generic Network.
template <typename T> class NetworkBase {
  public:
    NetworkBase() {}

    virtual ~NetworkBase() {}

    // Perform forward pass through the network
    virtual std::vector<sycl::event> forward_pass(const DeviceMatrix<T> &input,
                                                  DeviceMatrices<T> &intermediate_output_forward,
                                                  const std::vector<sycl::event> &deps) = 0;

    virtual std::vector<sycl::event> forward_pass(const DeviceMatrixView<T> &input,
                                                  DeviceMatricesView<T> intermediate_output_forward,
                                                  const std::vector<sycl::event> &deps) = 0;

    // Perform inference through the network
    virtual std::vector<sycl::event> inference(const DeviceMatrix<T> &input, DeviceMatrix<T> &output,
                                               const std::vector<sycl::event> &deps) = 0;

    virtual std::vector<sycl::event> inference(const DeviceMatrixView<T> &input, DeviceMatricesView<T> output,
                                               const std::vector<sycl::event> &deps) = 0;

    ///@brief input are the derivatives of the losses
    /// output are the updates of the weights for the optimization step.
    /// intermediate arrays are not used after this function
    virtual std::vector<sycl::event> backward_pass(const DeviceMatrix<T> &input, DeviceMatrices<T> &output,
                                                   DeviceMatrices<T> &intermediate_output_backward,
                                                   const DeviceMatrices<T> &intermediate_output_forward,
                                                   const std::vector<sycl::event> &deps,
                                                   DeviceMatrixView<T> *dL_dinput = nullptr) = 0;

    virtual std::vector<sycl::event> backward_pass(const DeviceMatrixView<T> input, DeviceMatricesView<T> output,
                                                   DeviceMatricesView<T> intermediate_output_backward,
                                                   const DeviceMatricesView<T> intermediate_output_forward,
                                                   const std::vector<sycl::event> &deps,
                                                   DeviceMatrixView<T> *dL_dinput = nullptr) = 0;
};

// Network Base class for all networks with weights matrices and width restrictions
template <typename T> class Network : public NetworkBase<T> {

  public:
    enum WeightInitMode { arange, constant_pos, constant_negative, xavier_normal, none };

    Network(sycl::queue &q, const int n_hidden_layers, const int input_width, const int network_width,
            const int output_width, const WeightInitMode mode)
        : m_q(q), input_width_(PadWidths(input_width, network_width)),
          output_width_(PadWidths(output_width, network_width)), original_input_width_(input_width),
          original_output_width_(output_width), n_hidden_layers_(NonNegative(n_hidden_layers)),
          network_width_(NonNegative(network_width)),
          m_weights_matrices(get_n_hidden_layers() + 1, get_input_width(), get_network_width(), get_network_width(),
                             get_network_width(), get_network_width(), get_output_width(), m_q),
          m_weightsT_matrices(get_n_hidden_layers() + 1, get_network_width(), get_input_width(), get_network_width(),
                              get_network_width(), get_output_width(), get_network_width(), m_q) {

        SanityCheck();
        initialize_weights_matrices(input_width, output_width, mode);
    }

    virtual ~Network() {}

    /// @brief Get the SYCL queue associated with the network
    sycl::queue &get_queue() { return m_q; }
    const sycl::queue &get_queue() const { return m_q; }

    virtual void set_weights_matrices(const std::vector<T> &weights) {
        m_weights_matrices.copy_from_host(weights).wait();
        ZeroWeightsPadding(original_input_width_, original_output_width_);
        TransposeWeights(m_weights_matrices, m_weightsT_matrices);
    };

    // this is the result from the backward pass. Not sure why this is here to be honest.
    virtual const DeviceMatrices<T> &get_weights_matrices() const { return m_weights_matrices; }
    virtual const DeviceMatrices<T> &get_weightsT_matrices() const { return m_weightsT_matrices; }

    virtual int get_n_hidden_layers() const { return n_hidden_layers_; }
    /// @brief returns hidden layers - 1
    /// @return n_hidden_layers - 1
    virtual int get_n_hidden_matrices() const { return n_hidden_layers_ - 1; }
    virtual int get_network_width() const { return network_width_; }
    virtual uint32_t get_input_width() const { return input_width_; }
    virtual uint32_t get_output_width() const { return output_width_; }
    virtual uint32_t get_unpadded_output_width() const { return original_output_width_; }
    virtual uint32_t get_inpadded_output_width() const { return original_input_width_; }

  private:
    virtual void SanityCheck() const {
        if (get_input_width() <= 0) {
            std::string errorMessage = "Input width of " + std::to_string(get_input_width()) +
                                       " is not supported. Value must be larger than 0.";
            throw std::runtime_error(errorMessage);
        }

        if (get_output_width() <= 0) {
            std::string errorMessage = "Output width of " + std::to_string(get_output_width()) +
                                       " is not supported. Value must be larger than 0.";
            throw std::runtime_error(errorMessage);
        }

        if (get_input_width() > network_width_) {
            std::string errorMessage = "Input width of " + std::to_string(get_input_width()) +
                                       " is not supported. Value must be <= network width (" +
                                       std::to_string(network_width_) + ").";
            throw std::runtime_error(errorMessage);
        }

        if (get_output_width() > network_width_) {
            std::string errorMessage = "Input width of " + std::to_string(get_output_width()) +
                                       " is not supported. Value must be <= network width (" +
                                       std::to_string(network_width_) + ").";
            throw std::runtime_error(errorMessage);
        }

        if (n_hidden_layers_ <= 0) {
            std::string errorMessage = "N hidden layers is " + std::to_string(output_width_) +
                                       " but must be >= 1, i.e., 1 hidden layer and 1 output layer.";
            throw std::runtime_error(errorMessage);
        }

        if (network_width_ != 16 && network_width_ != 32 && network_width_ != 64 && network_width_ != 128)
            throw std::invalid_argument("Network width has to be a power of 2 between 16 and 128.");

        if (network_width_ != get_input_width() || network_width_ != get_output_width())
            throw std::invalid_argument("Only networks with same input, layer and output width are allowed.");
    }

    ///@brief Helper function which sets values of the weights matrices to 0 if
    /// the actual input/output width was padded to the network-allowed input/output width.
    void ZeroWeightsPadding(const int unpadded_input_width, const int unpadded_output_width) {
        if (unpadded_input_width > get_input_width() || unpadded_output_width > get_output_width())
            throw std::invalid_argument("Padded weights width cannot be less than the unpadded.");

        /// we need to copy everything here since we do not want to have an implicit copy of 'this'
        const int padded_input_width = get_input_width();
        const int network_width = get_network_width();
        const int padded_output_width = get_output_width();

        // input matrix: set rows to 0.
        if (unpadded_input_width != padded_input_width) {
            DeviceMatrixView<T> weights = m_weights_matrices.Front();
            m_q.parallel_for(padded_input_width * network_width,
                             [=](auto idx) {
                                 const int i = idx / network_width; // rows
                                 const int j = idx % network_width; // cols

                                 if (i >= unpadded_input_width)
                                     weights.GetPointer()[toPackedLayoutCoord(i * network_width + j, padded_input_width,
                                                                              network_width)] = static_cast<T>(0);
                             })
                .wait();
        }

        // output matrix set columns to 0
        const int output_matrix_pos = m_weights_matrices.GetNumberOfMatrices() - 1;
        if (unpadded_output_width != padded_output_width) {
            DeviceMatrixView<T> weights = m_weights_matrices.Back();
            m_q.parallel_for(padded_output_width * network_width,
                             [=](auto idx) {
                                 const int i = idx / padded_output_width; // rows
                                 const int j = idx % padded_output_width; // cols

                                 if (j >= unpadded_output_width)
                                     weights.GetPointer()[toPackedLayoutCoord(i * padded_output_width + j,
                                                                              network_width, padded_output_width)] =
                                         static_cast<T>(0);
                             })
                .wait();
        }
    }

    void TransposeWeights(const DeviceMatrices<T> &weights, DeviceMatrices<T> &weightsT) {
        weights.PackedTranspose(weightsT);
        m_q.wait();
    }

    ///@brief initializes the weight matrices to pre-set values.
    /// Note that the network has in general no interest in knowing anything
    /// about padding. Only when we initialize the weight matrix are we setting
    /// certain elements to 0.
    ///@todo: remove this from the network class.
    void initialize_weights_matrices(const int unpadded_input_width, const int unpadded_output_width,
                                     WeightInitMode mode) {
        if (mode == WeightInitMode::arange) {
            throw std::invalid_argument("arange not supported");
        } else if (mode == WeightInitMode::constant_pos) {
            initialize_constant(m_weights_matrices, 0.01);
        } else if (mode == WeightInitMode::constant_negative) {
            initialize_constant(m_weights_matrices, -0.01);
        } else if (mode == WeightInitMode::xavier_normal) {
            initialize_xavier_normal(m_weights_matrices);

        } else if (mode == WeightInitMode::none) {
            initialize_constant(m_weights_matrices, 1.0);
        } else {
            throw std::invalid_argument("Invalid weights initialization mode.");
        }
        m_q.wait();

        ZeroWeightsPadding(unpadded_input_width, unpadded_output_width);

        TransposeWeights(m_weights_matrices, m_weightsT_matrices);
    }

    // Initialize memory with constant values
    static void initialize_constant(DeviceMatrices<T> &ms, const T &constant) { ms.fill(constant); }
    void initialize_xavier_normal(DeviceMatrices<T> &ms, const double weight_val_scaling_factor = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        double xavier_stddev = std::sqrt(2.0 / (network_width_ + network_width_));
        std::uniform_real_distribution<> dis(-weight_val_scaling_factor * xavier_stddev,
                                             weight_val_scaling_factor * xavier_stddev);
        std::vector<T> weight_matrix_vec(ms.nelements());

        for (T &val : weight_matrix_vec) {
            val = static_cast<T>(dis(gen));
        }
        ms.copy_from_host(weight_matrix_vec).wait();
    }

    static int PadWidths(const int width, const int base) {
        if (width <= 0) throw std::invalid_argument("width <= 0 cannot be padded.");
        if (base <= 0) throw std::invalid_argument("base <= 0 cannot be used for padding.");

        return ((width + base - 1) / base) * base;
    }

    static int NonNegative(const int number) {
        if (number < 0) throw std::invalid_argument("Use non-negative number.");

        return number;
    }

    sycl::queue &m_q;

    const uint32_t input_width_;
    const uint32_t output_width_;
    const uint32_t original_output_width_; // unpadded
    const uint32_t original_input_width_;  // unpadded
    const int n_hidden_layers_;
    const int network_width_;

    DeviceMatrices<T> m_weights_matrices;
    DeviceMatrices<T> m_weightsT_matrices;
};
