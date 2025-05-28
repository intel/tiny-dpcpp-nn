/**
 * @file tnn_api.h
 * @author Kai Yuan
 * @brief
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <c10/xpu/XPUStream.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "SwiftNetMLP.h"
#include "common.h"
#include "encoding_factory.h"
#include "io.h"
#include "ipex.h"
#include "json.hpp"
#include "oneapi/mkl.hpp"
#include "result_check.h"
#include "tensor_legacy_helper.h"

extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 16>;
extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 32>;
extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 64>;
extern template class SwiftNetMLP<sycl::ext::oneapi::bfloat16, 128>;

extern template class SwiftNetMLP<sycl::half, 16>;
extern template class SwiftNetMLP<sycl::half, 32>;
extern template class SwiftNetMLP<sycl::half, 64>;
extern template class SwiftNetMLP<sycl::half, 128>;

#define CHECK_XPU(x) \
  TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_XPU(x);        \
  CHECK_CONTIGUOUS(x)

using json = nlohmann::json;

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

namespace tnn {
class Module {
 public:
  Module() : sycl_queue_(InitQueue()) {}
  virtual ~Module() {}

  virtual torch::Tensor initialize_params() = 0;
  virtual torch::Tensor initialize_params(const torch::Tensor &tensor) = 0;
  virtual void set_params(const torch::Tensor &tensor,
                          bool weights_are_packed) = 0;
  virtual torch::Tensor get_params() = 0;
  virtual torch::Tensor forward_pass(const torch::Tensor &input_tensor) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor> backward_pass(
      const torch::Tensor &input_tensor, const torch::Tensor &input_from_fwd,
      bool pack_transpose_gradient, bool get_dl_dinput) = 0;
  virtual std::tuple<torch::Tensor, torch::Tensor> backward_pass(
      const torch::Tensor &input_tensor, bool pack_transpose_gradient,
      bool get_dl_dinput) = 0;
  virtual torch::Tensor inference(const torch::Tensor &input_tensor) = 0;
  virtual size_t n_params() = 0;
  virtual size_t n_output_dims() = 0;

  sycl::queue &get_queue() { return sycl_queue_; }
  // Conversion functions between dpcpp memory classes and torch::Tensors

  // Function to convert torch::Tensor to std::vector
  template <typename T>
  static std::vector<T> convertTensorToVector(const torch::Tensor &tensor) {
    auto tensor_cpu = tensor.to(torch::kCPU);
    // Convert the tensor directly to the target data type, this is only
    // supproted for float, double, and int
    return std::vector<T>(tensor_cpu.data_ptr<T>(),
                          tensor_cpu.data_ptr<T>() + tensor_cpu.numel());
  }

  // Specialization for bf16, which handles the bfloat16 tensors.
  template <>
  std::vector<bf16> convertTensorToVector<bf16>(const torch::Tensor &tensor) {
    // Move the tensor to the CPU if it's not already there, also need to
    // convert to float, as there's no data_ptr for bfloat16
    auto tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat32);

    const auto *begin = tensor_cpu.data_ptr<float>();
    const auto *end = begin + tensor_cpu.numel();

    std::vector<bf16> result;
    result.reserve(tensor_cpu.numel());
    for (const auto *it = begin; it != end; ++it) {
      result.push_back(bf16{*it});
    }
    return result;
  }

  // Specialization for fp16
  template <>
  std::vector<fp16> convertTensorToVector<fp16>(const torch::Tensor &tensor) {
    auto tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat32);

    const auto *begin = tensor_cpu.data_ptr<float>();
    const auto *end = begin + tensor_cpu.numel();

    std::vector<fp16> result;
    result.reserve(tensor_cpu.numel());

    for (const auto *it = begin; it != end; ++it) {
      result.push_back(fp16{*it});
    }

    return result;
  }

  template <typename T>
  static torch::Tensor convertVectorToTensor(const std::vector<T> &v) {
    const torch::TensorOptions &options =
        torch::TensorOptions().dtype(torch_type<T>::dtype).device(torch::kCPU);

    // Use const_cast to cast away constness and create a non-const void*
    // pointer
    void *data = const_cast<void *>(static_cast<const void *>(v.data()));

    // Create a tensor from the vector's data, with the same size and provided
    // options. It's important to call clone because from_blob does not take
    // ownership of the memory.
    torch::Tensor tensor =
        torch::from_blob(data, {static_cast<int64_t>(v.size())}, options)
            .clone();
    return tensor.to(torch::kXPU);
  }

  template <>
  torch::Tensor convertVectorToTensor<bf16>(const std::vector<bf16> &v) {
    const torch::TensorOptions &options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    std::vector<float> float_vector;
    float_vector.reserve(v.size());  // Reserve space to improve performance
    for (const bf16 &value : v) {
      float_vector.push_back(static_cast<float>(value));
    }

    // Create a tensor from the vector's data, with the same size and provided
    // options. It's important to call clone because from_blob does not take
    // ownership of the memory.
    torch::Tensor tensor =
        torch::from_blob(float_vector.data(),
                         {static_cast<int64_t>(float_vector.size())}, options)
            .clone();

    return tensor.to(torch::kXPU);
  }

  template <>
  torch::Tensor convertVectorToTensor<fp16>(const std::vector<fp16> &v) {
    const torch::TensorOptions &options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    std::vector<float> float_vector;
    float_vector.reserve(v.size());  // Reserve space to improve performance
    for (const fp16 &value : v) {
      float_vector.push_back(static_cast<float>(value));
    }

    // Create a tensor from the vector's data, with the same size and provided
    // options. It's important to call clone because from_blob does not take
    // ownership of the memory.
    torch::Tensor tensor =
        torch::from_blob(float_vector.data(),
                         {static_cast<int64_t>(float_vector.size())}, options)
            .clone();

    return tensor.to(torch::kXPU);
  }

  // Function to convert a device pointer to a torch tensor on the device
  template <typename T>
  static torch::Tensor convertDeviceMatrixViewToTorchTensor(
      const DeviceMatrixView<T> &dm) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, bf16>::value ||
                      std::is_same<T, fp16>::value,
                  "Unsupported data type. Only float and sycl::half are "
                  "supported. More needs to be manually "
                  "defined in tnn_api.h");
    // Create a torch::Tensor from the data_ptr
    torch::Tensor tensor = xpu::dpcpp::fromUSM(
        static_cast<T *>(dm.GetPointer()), torch_type<T>::dtype,
        {static_cast<long>(dm.m()), static_cast<long>(dm.n())});
    CHECK_INPUT(tensor);
    return tensor;
  }

  // Function to convert a device pointer to a torch tensor on the device
  template <typename T>
  static torch::Tensor convertDeviceMatrixToTorchTensor(DeviceMatrix<T> &dm) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, bf16>::value ||
                      std::is_same<T, fp16>::value,
                  "Unsupported data type. Only float and sycl::half are "
                  "supported. More needs to be manually "
                  "defined in tnn_api.h");
    // Create a torch::Tensor from the data_ptr
    torch::Tensor tensor = xpu::dpcpp::fromUSM(
        dm.data(), torch_type<T>::dtype,
        {static_cast<long>(dm.rows()), static_cast<long>(dm.cols())});
    CHECK_INPUT(tensor);
    return tensor;
  }

  // Function to convert a device pointer to a torch tensor on the device
  template <typename T>
  static torch::Tensor convertDeviceMatricesToTorchTensor(
      const DeviceMatrices<T> &dm) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, bf16>::value ||
                      std::is_same<T, fp16>::value,
                  "Unsupported data type. Only float and sycl::half are "
                  "supported. More needs to be manually "
                  "defined in tnn_api.h");
    // Create a torch::Tensor from the data_ptr
    torch::Tensor tensor = xpu::dpcpp::fromUSM(
        dm.GetViews().GetMatrixPointer(0), torch_type<T>::dtype,
        {static_cast<long>(dm.nelements()), static_cast<long>(1)});
    CHECK_INPUT(tensor);
    return tensor;
  }

  // Function to convert a device pointer to a torch tensor on the device
  template <typename T>
  static torch::Tensor convertDeviceMemToTorchTensor(DeviceMem<T> &dm) {
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, bf16>::value ||
                      std::is_same<T, fp16>::value,
                  "Unsupported data type. Only float and sycl::half are "
                  "supported. More needs to be manually "
                  "defined in tnn_api.h");
    // Create a torch::Tensor from the data_ptr
    torch::Tensor tensor = xpu::dpcpp::fromUSM(
        dm.data(), torch_type<T>::dtype,
        {static_cast<long>(dm.size()), static_cast<long>(1)});
    CHECK_INPUT(tensor);
    return tensor;
  }

 protected:
  virtual void FreeWorkspace() = 0;
  virtual void AllocateWorkspace(const size_t batch_size) = 0;
  // A utility type trait to map C++ types to PyTorch data types
  template <typename T>
  struct torch_type {};

  template <>
  struct torch_type<int> {
    static const auto dtype = torch::kInt32;
  };

  template <>
  struct torch_type<double> {
    static const auto dtype = torch::kFloat64;
  };

  template <>
  struct torch_type<float> {
    static const auto dtype = torch::kFloat32;
  };

  template <>
  struct torch_type<fp16> {
    static const auto dtype = c10::ScalarType::Half;
  };

  template <>
  struct torch_type<bf16> {
    static const auto dtype = c10::ScalarType::BFloat16;
  };

 private:
  sycl::queue &sycl_queue_;
  static sycl::queue &InitQueue() {
    return c10::xpu::getCurrentXPUStream().queue();
    // auto device_type = c10::DeviceType::XPU;
    // c10::impl::VirtualGuardImpl impl(device_type);
    // c10::Stream xpu_stream = impl.getStream(c10::Device(device_type));
    // return xpu::get_queue_from_stream(xpu_stream);
  }
};

/************************************************ENCODING MODULE */
template <typename T>
class EncodingModule : public Module {
 public:
  EncodingModule(std::string filename,
                 std::optional<uint32_t> padded_output_width = std::nullopt)
      : max_batch_size_(0), output_encoding_ptr_(nullptr) {
    json encoding_config = io::loadJsonConfig(filename);
    encoding_ = create_encoding<T>(encoding_config, this->get_queue(),
                                   padded_output_width);
    initialize_ptr();
  }

  EncodingModule(const json &encoding_config,
                 std::optional<uint32_t> padded_output_width = std::nullopt)
      : max_batch_size_(0), output_encoding_ptr_(nullptr) {
    const json encoding_config_validated =
        io::validateAndCopyEncodingConfig(encoding_config);

    encoding_ = create_encoding<T>(encoding_config_validated, this->get_queue(),
                                   padded_output_width);

    initialize_ptr();
  }

  ~EncodingModule() { this->FreeWorkspace(); }

  torch::Tensor forward_pass(const torch::Tensor &input_tensor) override {
    CHECK_INPUT(input_tensor);

    const size_t batch_size = input_tensor.sizes()[0];
    if (input_tensor.sizes().size() != 2 ||
        input_tensor.sizes()[1] != encoding_->get_input_width()) {
      throw std::runtime_error("Input tensor has incorrect dimensions");
    }

    DeviceMatrixView<T> input_dmv(
        batch_size, encoding_->get_input_width(), encoding_->get_input_width(),
        reinterpret_cast<T *>(input_tensor.data_ptr()));
    this->AllocateWorkspace(batch_size);
    DeviceMatrixView<T> output_encoding(
        batch_size, encoding_->get_padded_output_width(),
        encoding_->get_padded_output_width(), output_encoding_ptr_);
    std::unique_ptr<Context> model_ctx =
        encoding_->forward_impl(input_dmv, &output_encoding);
    this->get_queue().wait();

    return xpu::dpcpp::fromUSM(
        (output_encoding.GetPointer()), torch_type<T>::dtype,
        {static_cast<long long>(batch_size),
         static_cast<long long>(encoding_->get_padded_output_width())});
  }

  torch::Tensor inference(const torch::Tensor &input_tensor) override {
    return forward_pass(input_tensor);
  }

  std::tuple<torch::Tensor, torch::Tensor> backward_pass(
      const torch::Tensor &grad_output, bool pack_transpose_gradient = false,
      bool get_dl_dinput = false) override {
    throw std::runtime_error("backward_pass needs input_from_fwd");
  }

  std::tuple<torch::Tensor, torch::Tensor> backward_pass(
      const torch::Tensor &grad_output, const torch::Tensor &input_from_fwd,
      bool pack_transpose_gradient = false,
      bool get_dl_dinput = false) override {
    if (encoding_->get_n_params() == 0) {
      return {torch::Tensor(), torch::Tensor()};
    }

    CHECK_INPUT(grad_output);

    if (pack_transpose_gradient) {
      throw std::invalid_argument("Encoding params are not packed");
    }
    if (get_dl_dinput) {
      throw std::invalid_argument("Encoding get_dl_dinput not implemented");
    }

    std::unique_ptr<Context> model_ctx = nullptr;
    const auto batch_size = grad_output.sizes()[0];
    const int64_t input_width = encoding_->get_input_width();

    // Assuming input_from_fwd should have dimensions [batch_size,
    // input_width*input_width]
    const auto input_sizes = input_from_fwd.sizes();
    if (input_sizes.size() != 2 || input_sizes[0] != batch_size ||
        input_sizes[1] != input_width) {
      // Sizes do not match, throw an exception
      throw std::runtime_error("Size mismatch for input_from_fwd: expected [" +
                               std::to_string(batch_size) + ", " +
                               std::to_string(input_width * input_width) +
                               "], but got [" + std::to_string(input_sizes[0]) +
                               ", " + std::to_string(input_sizes[1]) + "]");
    }

    DeviceMatrixView<T> dL_doutput_dmv(
        batch_size, encoding_->get_padded_output_width(),
        encoding_->get_padded_output_width(),
        reinterpret_cast<T *>(grad_output.data_ptr()));

    DeviceMatrixView<T> input_encoding_dmv(
        batch_size, encoding_->get_input_width(), encoding_->get_input_width(),
        reinterpret_cast<T *>(input_from_fwd.data_ptr()));

    auto gradients_view = gradients_->GetView();
    encoding_->backward_impl(*model_ctx, input_encoding_dmv, dL_doutput_dmv,
                             &gradients_view);
    this->get_queue().wait();
    return {torch::Tensor(), convertDeviceMatrixToTorchTensor(*gradients_)};
  }

  torch::Tensor initialize_params() override { return get_params(); }

  torch::Tensor initialize_params(const torch::Tensor &tensor) override {
    set_params(tensor, false);
    return get_params();
  }

  void set_params(const torch::Tensor &params,
                  bool weights_are_packed) override {
    if (encoding_->get_params() == nullptr) {
      throw std::runtime_error("no encoding params to set");
    }
    if (weights_are_packed)
      throw std::runtime_error(
          "Packed weights are not supported in api set params");
    encoding_->get_params()->copy_from_device(params.data_ptr<T>());
  }

  torch::Tensor get_params() override {
    if (encoding_->get_params()) {
      return convertDeviceMatrixToTorchTensor<T>(*encoding_->get_params());
    } else {
      return torch::empty({0});
    }
  }

  size_t n_params() override { return encoding_->get_n_params(); }
  size_t n_output_dims() override {
    return encoding_->get_padded_output_width();
  }

  std::vector<T> get_encoding_params() {
    return encoding_->get_params()->copy_to_host();
  }
  std::vector<T> get_encoding_grads() { return gradients_->copy_to_host(); }

 protected:
  void FreeWorkspace() override {
    sycl::free(output_encoding_ptr_, this->get_queue());
    output_encoding_ptr_ = nullptr;
    max_batch_size_ = 0;
  }

  void AllocateWorkspace(const size_t batch_size) override {
    if (batch_size > max_batch_size_) {
      this->FreeWorkspace();
      const size_t n_elems =
          batch_size * (encoding_->get_padded_output_width());
      output_encoding_ptr_ = sycl::malloc_device<T>(n_elems, this->get_queue());
      max_batch_size_ = batch_size;
      this->get_queue().wait();
    }
  }

 private:
  void initialize_ptr() {
    if (encoding_ && encoding_->get_n_params()) {
      gradients_ = std::make_unique<DeviceMatrix<T>>(encoding_->get_n_params(),
                                                     1, this->get_queue());
      gradients_->fill(0.0f).wait();
    }
  }

  std::shared_ptr<Encoding<T>> encoding_;
  std::unique_ptr<DeviceMatrix<T>> gradients_;

  // dependent on the batch size. We allocate enough such that max_batch_size
  // would fit.
  size_t max_batch_size_;
  T *output_encoding_ptr_;
};

/**************************************NETWORK MODULE */
template <typename T, int WIDTH>
class NetworkModule : public Module {
 public:
  NetworkModule(const int input_width, const int output_width,
                const int n_hidden_layers, const Activation activation,
                const Activation output_activation, bool use_bias)
      : network_(this->get_queue(), input_width, output_width, n_hidden_layers,
                 activation, output_activation,
                 Network<T>::WeightInitMode::xavier_normal, use_bias),
        net_gradients_(network_.get_n_hidden_layers() + 1,
                       network_.get_network_width(), WIDTH, WIDTH, WIDTH, WIDTH,
                       network_.get_output_width(), this->get_queue()),
        max_batch_size_(0),
        interm_forw_ptr_(nullptr),
        interm_backw_ptr_(nullptr),
        dL_dinput_ptr_(nullptr) {
    net_gradients_.fill(0.0).wait();
  }

  ~NetworkModule() { this->FreeWorkspace(); }

  torch::Tensor inference(const torch::Tensor &input_tensor) override {
    CHECK_INPUT(input_tensor);

    const size_t batch_size = input_tensor.sizes()[0];
    DeviceMatrixView<T> input_dm(
        batch_size, network_.get_input_width(), network_.get_input_width(),
        reinterpret_cast<T *>(input_tensor.data_ptr()));
    this->AllocateWorkspace(batch_size);

    DeviceMatricesView<T> output_network(1, batch_size,
                                         network_.get_output_width(), 0, 0, 0,
                                         0, interm_forw_ptr_);
    network_.inference(input_dm, output_network, {});
    this->get_queue().wait();
    return xpu::dpcpp::fromUSM(
        (interm_forw_ptr_), torch_type<T>::dtype,
        {static_cast<long>(batch_size),
         static_cast<long>(network_.get_output_width())});
  }

  torch::Tensor forward_pass(const torch::Tensor &input_tensor) override {
    CHECK_INPUT(input_tensor);

    const size_t batch_size = input_tensor.sizes()[0];
    DeviceMatrixView<T> input_dm(
        batch_size, network_.get_input_width(), network_.get_input_width(),
        reinterpret_cast<T *>(input_tensor.data_ptr()));
    this->AllocateWorkspace(batch_size);

    DeviceMatricesView<T> output_network(
        network_.get_n_hidden_layers() + 2, batch_size,
        network_.get_input_width(), batch_size, network_.get_network_width(),
        batch_size, network_.get_output_width(), interm_forw_ptr_);
    network_.forward_pass(input_dm, output_network, {});
    this->get_queue().wait();
    return xpu::dpcpp::fromUSM(
        (output_network.GetMatrixPointer(network_.get_n_hidden_layers() + 1)),
        torch_type<T>::dtype,
        {static_cast<long>(batch_size),
         static_cast<long>(network_.get_output_width())});
  }

  std::tuple<torch::Tensor, torch::Tensor> backward_pass(
      const torch::Tensor &grad_output, const torch::Tensor &input_from_fwd,
      bool pack_transpose_gradient, bool get_dl_dinput) override {
    throw std::runtime_error(
        "backward_pass of NetworkModule takes no input_from_fwd");
  }

  std::tuple<torch::Tensor, torch::Tensor> backward_pass(
      const torch::Tensor &grad_output, bool pack_transpose_gradient,
      bool get_dl_dinput) override {
    CHECK_INPUT(grad_output);

    const int batch_size = grad_output.sizes()[0];
    if (grad_output.size(1) != network_.get_output_width()) {
      throw std::invalid_argument(
          "grad_output.size(1): " + std::to_string(grad_output.size(1)) +
          " is not equal to  get_padded_output_width" +
          std::to_string(network_.get_output_width()));
    }

    DeviceMatrixView<T> dL_doutput(
        batch_size, network_.get_output_width(), network_.get_output_width(),
        reinterpret_cast<T *>(grad_output.data_ptr()));

    this->AllocateWorkspace(batch_size);
    DeviceMatricesView<T> interm_bwd(
        network_.get_n_hidden_layers() + 1, batch_size,
        network_.get_input_width(), batch_size, network_.get_network_width(),
        batch_size, network_.get_output_width(), interm_backw_ptr_);
    DeviceMatricesView<T> interm_fwd(
        network_.get_n_hidden_layers() + 2, batch_size,
        network_.get_input_width(), batch_size, network_.get_network_width(),
        batch_size, network_.get_output_width(), interm_forw_ptr_);

    DeviceMatrixView<T> *dL_dinput = nullptr;
    if (get_dl_dinput) {
      dL_dinput =
          new DeviceMatrixView<T>(batch_size, network_.get_input_width(),
                                  network_.get_input_width(), dL_dinput_ptr_);
    }

    network_.backward_pass(dL_doutput, net_gradients_.GetViews(), interm_bwd,
                           interm_fwd, {}, dL_dinput);
    this->get_queue().wait();

    if (pack_transpose_gradient) {
      // Pack and transpose gradient as we multiply from left (Wx) instead of
      // from the right as in pytorch
      net_gradients_.PackAndTranspose(net_gradients_);
    }

    torch::Tensor input_grad;
    if (get_dl_dinput) {
      input_grad = xpu::dpcpp::fromUSM(
          (dL_dinput->GetPointer()), torch_type<T>::dtype,
          {static_cast<long>(dL_dinput->nelements()), static_cast<long>(1)});
      delete dL_dinput;  // delete only the view (underlying pointer still
                         // managed by Module class)
      dL_dinput = nullptr;
    }

    return {input_grad,
            xpu::dpcpp::fromUSM((net_gradients_.GetViews().GetMatrixPointer(0)),
                                torch_type<T>::dtype,
                                {static_cast<long>(net_gradients_.nelements()),
                                 static_cast<long>(1)})};
  }

  torch::Tensor initialize_params() override { return get_params(); }

  torch::Tensor get_params() override {
    return convertDeviceMatricesToTorchTensor(network_.get_weights_matrices());
  }
  torch::Tensor initialize_params(const torch::Tensor &tensor) override {
    set_params(tensor, false);
    return initialize_params();
  }

  size_t n_params() override {
    return network_.get_weights_matrices().nelements();
  }
  size_t n_output_dims() override { return network_.get_output_width(); }

  void set_params(const torch::Tensor &params,
                  bool weights_are_packed) override {
    network_.set_weights_matrices(convertTensorToVector<T>(params),
                                  weights_are_packed);
  }

  std::vector<T> get_network_params() {
    return network_.get_weights_matrices().copy_to_host();
  }
  std::vector<T> get_network_grads() { return net_gradients_.copy_to_host(); }

 protected:
  void AllocateWorkspace(const size_t batch_size) override {
    if (batch_size > max_batch_size_) {
      this->FreeWorkspace();
      const size_t n_elems_forw =
          batch_size *
          ((size_t)network_.get_input_width() + network_.get_output_width() +
           network_.get_network_width() * network_.get_n_hidden_layers());
      interm_forw_ptr_ =
          sycl::malloc_device<T>(n_elems_forw, this->get_queue());

      const size_t n_elems_backw =
          batch_size *
          ((size_t)network_.get_output_width() +
           network_.get_network_width() * network_.get_n_hidden_layers());
      interm_backw_ptr_ =
          sycl::malloc_device<T>(n_elems_backw, this->get_queue());

      const size_t n_elems_dldinput =
          batch_size * (size_t)network_.get_input_width();
      dL_dinput_ptr_ =
          sycl::malloc_device<T>(n_elems_dldinput, this->get_queue());

      max_batch_size_ = batch_size;
      this->get_queue().wait();
    }
  }

  void FreeWorkspace() override {
    sycl::free(interm_forw_ptr_, this->get_queue());
    interm_forw_ptr_ = nullptr;
    sycl::free(interm_backw_ptr_, this->get_queue());
    interm_backw_ptr_ = nullptr;
    sycl::free(dL_dinput_ptr_, this->get_queue());
    dL_dinput_ptr_ = nullptr;
    max_batch_size_ = 0;
  }

 private:
  SwiftNetMLP<T, WIDTH> network_;
  // independent of the batch_size
  DeviceMatrices<T> net_gradients_;

  // dependent on the batch size. We allocate enough such that max_batch_size
  // would fit.
  size_t max_batch_size_;
  T *interm_forw_ptr_;
  T *interm_backw_ptr_;
  T *dL_dinput_ptr_;
};

}  // namespace tnn
