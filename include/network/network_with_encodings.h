/**
 * @file network_with_encodings.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a network with an encoding.
 * TODO: somehow consolidate this as a type of network. Requires to rethink our network class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "Network.h"
#include "SwiftNetMLP.h"
#include "encoding_factory.h"
#include "result_check.h"

template <typename T_src, typename T_dst>
DeviceMatrixView<T_dst> CreateAndConvertToDeviceMatrixView(const DeviceMatrixView<T_src> &srcView, sycl::queue &queue) {

    // Allocate memory for the new view on the device.
    T_dst *dstData = sycl::malloc_device<T_dst>(srcView.nelements(), queue);

    // Create a new DeviceMatrixView for T_dst with the same dimensions as the source view.
    DeviceMatrixView<T_dst> dstView(srcView.m(), srcView.n(), srcView.n(), dstData);

    // Make sure the dimensions of the source and destination views match.
    if (srcView.m() != dstView.m() || srcView.n() != dstView.n()) {
        throw std::invalid_argument("Source and destination views must have the same dimensions.");
    }

    // Get the pointer to the source data.
    T_src *srcData = srcView.GetPointer();

    // Enqueue a parallel_for to convert each element from T_src to T_dst.
    queue
        .parallel_for(sycl::range<1>(srcView.nelements()),
                      [=](sycl::id<1> idx) { dstData[idx] = static_cast<T_dst>(srcData[idx]); })
        .wait(); // Synchronization can be omitted for asynchronous execution if required.

    // Return the new view with converted data.
    return dstView;
}
template <typename T_net, typename T_enc>
void CopyToDeviceMem(const DeviceMatricesView<T_net> net_grad, DeviceMem<T_enc> &devicemem, sycl::queue &q) {

    // Enqueue a kernel to typecast and copy net_grad to devicemem
    q.submit([&](sycl::handler &h) {
         // Obtain a DeviceMatrixView for the entire net_grad data
         // Get the raw pointers for source and destination
         T_net *net_grad_ptr = net_grad.GetMatrixPointer(0);
         T_enc *devicemem_ptr = devicemem.data();

         h.parallel_for(sycl::range<1>(net_grad.nelements()), [=](sycl::id<1> idx) {
             // Typecast each element of net_grad and store into devicemem
             devicemem_ptr[idx] = static_cast<T_enc>(net_grad_ptr[idx]);
         });
     }).wait();
}

template <typename T_net, typename T_enc>
void CopyToDeviceMem(const DeviceMatricesView<T_net> net_grad, DeviceMatrixView<T_enc> enc_grad,
                     DeviceMem<T_enc> &devicemem, sycl::queue &q) {
    CopyToDeviceMem<T_net, T_enc>(net_grad, devicemem, q);
    // Copy enc_grad to the end of the content from devicemem using memcpy by calculating the offset pointer
    T_enc *offset_ptr = devicemem.data() + net_grad.nelements();
    q.memcpy(offset_ptr, enc_grad.GetPointer(), enc_grad.nelements() * sizeof(T_enc)).wait();
}

template <typename SrcType, typename DstType>
DeviceMatrix<DstType> convert_matrix(const DeviceMatrix<SrcType> &src, sycl::queue &q) {
    // Create a new DeviceMatrix of the destination type with the
    // same dimensions as the source matrix.
    DeviceMatrix<DstType> dest(src.rows(), src.cols(), q);

    // Get the pointers to the underlying data of the source and destination matrices.
    SrcType const *src_data = src.data();
    DstType *dest_data = dest.data();

    // Compute the number of elements to convert.
    size_t num_elements = src.size();

    // Launch the SYCL kernel to perform the conversion.
    q.parallel_for(num_elements, [=](sycl::id<1> idx) {
         dest_data[idx] = static_cast<DstType>(src_data[idx]); // Conversion from SrcType to DstType
     }).wait(); // Wait for the kernel to complete before returning the new matrix.

    return dest;
}

template <typename T_enc, typename T_net> class NetworkWithEncoding {
  public:
    NetworkWithEncoding() = delete;

    NetworkWithEncoding(std::shared_ptr<Encoding<T_enc>> encoding, std::shared_ptr<Network<T_net>> network)
        : encoding_(encoding), network_(network) {
        SanityCheck();
    }

    ~NetworkWithEncoding() {}

    std::vector<sycl::event> inference(const DeviceMatrixView<float> input, DeviceMatrix<T_net> &network_input,
                                       DeviceMatrix<T_enc> &encoding_output, DeviceMatricesView<T_net> network_output,
                                       const std::vector<sycl::event> &deps) {
        /// TODO: implemente proper usage of deps. Requires proper implementation of forward_impl
        /// in encodings which takes it as input and returns new dependencies.

        const int batch_size = input.m();
        const int network_input_width = network_->get_input_width();

        if (network_input.m() != batch_size || encoding_output.m() != batch_size)
            throw std::invalid_argument("Wrong dimensions.");
        if (encoding_output.n() != network_input_width) throw std::invalid_argument("Wrong dimensions.");

        auto encoding_output_view = encoding_output.GetView();
        auto ctxt = encoding_->forward_impl(&network_->get_queue(), input, &encoding_output_view);
        network_->get_queue().wait();

        network_input.copy_from_device(encoding_output.data());
        network_->get_queue().wait();
        auto network_input_view = network_input.GetView();
        network_->inference(network_input_view, network_output, {});
        return {};
    }

    std::vector<sycl::event> forward_pass(const DeviceMatrixView<float> input, DeviceMatrix<T_net> &network_input,
                                          DeviceMatrix<T_enc> &encoding_output,
                                          DeviceMatricesView<T_net> intermediate_forward,
                                          const std::vector<sycl::event> &deps) {
        /// TODO: implemente proper usage of deps. Requires proper implementation of forward_impl
        /// in encodings which takes it as input and returns new dependencies.

        const int batch_size = input.m();
        const int network_input_width = network_->get_input_width();

        if (network_input.m() != batch_size || encoding_output.m() != batch_size)
            throw std::invalid_argument("Wrong dimensions."); /// TODO: need more asserts here
        if (encoding_output.n() != network_input_width) throw std::invalid_argument("Wrong dimensions.");

        auto encoding_output_view = encoding_output.GetView();
        auto ctxt = encoding_->forward_impl(&network_->get_queue(), input, &encoding_output_view);

        network_->get_queue().wait();

        network_input.copy_from_device(encoding_output.data());

        network_->get_queue().wait();

        const DeviceMatrixView<T_net> network_input_view = network_input.GetView();
        return network_->forward_pass(network_input_view, intermediate_forward, {});
    }

    std::vector<sycl::event>
    backward_pass(const DeviceMatrixView<T_net> input_backward, DeviceMatricesView<T_net> network_gradient,
                  DeviceMatricesView<T_net> intermediate_backward, const DeviceMatricesView<T_net> intermediate_forward,
                  const std::vector<sycl::event> &deps, DeviceMatrixView<T_net> *dL_dinput = nullptr) {
        // this backward_pass function does not have encoding params
        std::vector<sycl::event> event;
        event = network_->backward_pass(input_backward, network_gradient, intermediate_backward, intermediate_forward,
                                        deps, dL_dinput);
        network_->get_queue().wait();
        return event;
    }

    std::vector<sycl::event>
    backward_pass(const DeviceMatrixView<T_net> input_backward, DeviceMatricesView<T_net> net_gradients,
                  DeviceMatrixView<T_enc> enc_gradients, DeviceMatricesView<T_net> intermediate_backward,
                  const DeviceMatricesView<T_net> intermediate_forward, const std::vector<sycl::event> &deps,
                  const DeviceMatrixView<T_enc> input_encoding, DeviceMatrixView<T_net> *dL_dinput) {
        // this function has encoding params, i.e., grid encoding
        if (!encoding_->n_params()) {
            throw std::runtime_error("This backward pass implementation requires encoding to have params for which "
                                     "gradients can be calculated.");
        }
        std::vector<sycl::event> event = network_->backward_pass(input_backward, net_gradients, intermediate_backward,
                                                                 intermediate_forward, deps, dL_dinput);
        network_->get_queue().wait();
        /// TODO: resolve events, we already wait
        const int batch_size = input_backward.m();
        std::unique_ptr<Context> model_ctx = nullptr;

        DeviceMatrixView<T_enc> dL_dinput_float =
            CreateAndConvertToDeviceMatrixView<T_net, T_enc>(*dL_dinput, network_->get_queue());
        encoding_->backward_impl(&network_->get_queue(), *model_ctx, input_encoding, dL_dinput_float, &enc_gradients);
        network_->get_queue().wait();
        return {};
    }

    // functions which simplify the usage by generating the intermediate arrays
    DeviceMatrix<T_net> GenerateIntermediateForwardMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_input_width() +
                                    network_->get_network_width() * network_->get_n_hidden_layers() +
                                    network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(batch_size, tmp_n_cols, network_->get_queue()));
    }

    DeviceMatrix<T_net> GenerateIntermediateBackwardMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols =
            network_->get_network_width() * network_->get_n_hidden_layers() + network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T_enc> GenerateEncodingOutputMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_input_width();
        return std::move(DeviceMatrix<T_enc>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T_net> GenerateForwardOutputMatrix(const size_t batch_size) {
        const uint32_t tmp_n_cols = network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(batch_size, tmp_n_cols, network_->get_queue()));
    }
    DeviceMatrix<T_net> GenerateBackwardOutputMatrix() {
        const uint32_t tmp_n_rows = network_->get_network_width();
        const uint32_t tmp_n_cols = network_->get_n_hidden_matrices() * network_->get_network_width() +
                                    network_->get_input_width() + network_->get_output_width();
        return std::move(DeviceMatrix<T_net>(tmp_n_rows, tmp_n_cols, network_->get_queue()));
    }

    std::shared_ptr<Encoding<T_enc>> get_encoding() { return encoding_; }

    std::shared_ptr<Network<T_net>> get_network() { return network_; }

    void set_network_params(std::vector<T_net> network_params) { network_->set_weights_matrices(network_params); }

    void set_encoding_params(DeviceMatrix<T_enc> &encoding_params_dev_matrix,
                             std::vector<T_enc> *encoding_params = nullptr) {
        encoding_->set_params(encoding_params_dev_matrix, encoding_params);
    }

    void set_encoding_params(std::vector<T_enc> params) { encoding_->set_params(params); }

    void initialize_params(DeviceMatrix<T_enc> &encoding_params_dev_matrix,
                           std::vector<T_enc> *encoding_params = nullptr,
                           std::vector<T_net> *network_params = nullptr) {

        set_encoding_params(encoding_params_dev_matrix, encoding_params);
        if (network_params != nullptr) {
            set_network_params(*network_params);
        }
    }

  private:
    void SanityCheck() const {
        /// TODO: check that the queues of the encoding and network coincide.

        if (encoding_->padded_output_width() != network_->get_input_width())
            throw std::invalid_argument("Encoding output dim and network input dim mismatch. Expected: " +
                                        std::to_string(encoding_->padded_output_width()) +
                                        " for encoding padded output width, but got network input width: " +
                                        std::to_string(network_->get_input_width()));
    }

    std::shared_ptr<Encoding<T_enc>> encoding_;
    std::shared_ptr<Network<T_net>> network_;
};

template <typename T_enc, typename T_net, int WIDTH>
std::shared_ptr<NetworkWithEncoding<T_enc, T_net>>
create_network_with_encoding(sycl::queue &q, const int output_width, const int n_hidden_layers, Activation activation,
                             Activation output_activation, const json &encoding_config) {
    // input width is encoding_config as EncodingParams::N_DIMS_TO_ENCODE
    std::shared_ptr<Encoding<T_enc>> enc = create_encoding<T_enc>(encoding_config);
    if (enc->output_width() < WIDTH) {
        enc->set_padded_output_width(WIDTH);
    }
    std::shared_ptr<SwiftNetMLP<T_net, WIDTH>> net = std::make_shared<SwiftNetMLP<T_net, WIDTH>>(
        q, WIDTH, output_width, n_hidden_layers, activation, output_activation);
    return std::make_shared<NetworkWithEncoding<T_enc, T_net>>(enc, net);
}
