/**
 * @file doctest_torch_memory.cpp
 * @author Kai Yuan (kai.yuan@intel.com)
 * @brief Tests for the Torch and tnn_api.h.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"
#include "tnn_api.h"
#include <ipex.h>

template <typename T> void FillAndTestDeviceMemory(const std::string &test_case_name, sycl::queue &queue) {
    SUBCASE((test_case_name + " - DeviceMatrix").c_str()) {
        DeviceMatrix<T> dm(16, 8, queue);
        dm.fill(static_cast<T>(2.56)).wait();
        torch::Tensor tensor = tnn::Module::convertDeviceMatrixToTorchTensor(dm);
        CHECK(tensor.device().is_xpu());

        torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat64);
        CHECK(tensor_cpu.device().is_cpu());
        auto result_vector = dm.copy_to_host();
        auto data_ptr = tensor_cpu.data_ptr<double>();

        for (size_t i = 0; i < result_vector.size(); ++i) {
            CHECK(doctest::Approx(data_ptr[i]) == static_cast<double>(result_vector[i]));
        }
    }

    SUBCASE((test_case_name + " - DeviceMatrixView").c_str()) {
        DeviceMatrix<T> dm(16, 1, queue);
        dm.fill(static_cast<T>(2.56f)).wait();
        DeviceMatrixView<T> dm_view = dm.GetView();
        torch::Tensor tensor = tnn::Module::convertDeviceMatrixViewToTorchTensor<T>(dm_view);

        CHECK(tensor.device().is_xpu());

        torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat64);
        CHECK(tensor_cpu.device().is_cpu());
        auto result_vector = dm.copy_to_host();
        auto data_ptr = tensor_cpu.data_ptr<double>();

        for (size_t i = 0; i < result_vector.size(); ++i) {
            CHECK(doctest::Approx(data_ptr[i]) == static_cast<double>(result_vector[i]));
        }
    }

    SUBCASE((test_case_name + " - DeviceMatrixView as back").c_str()) {
        DeviceMatrices<T> dm(4, 8, 16, 16, 16, 16, 16, queue);
        dm.fill(static_cast<T>(2.56f)).wait();
        DeviceMatrixView<T> dm_view = dm.Back();
        torch::Tensor tensor = tnn::Module::convertDeviceMatrixViewToTorchTensor<T>(dm_view);
        CHECK(tensor.device().is_xpu());

        torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat64);
        CHECK(tensor_cpu.device().is_cpu());
        auto result_vector = dm.copy_to_host();
        std::vector<T> last_elements(result_vector.end() - 16 * 16, result_vector.end());

        CHECK(last_elements.size() == 16 * 16);
        auto data_ptr = tensor_cpu.data_ptr<double>();
        CHECK(last_elements.size() == tensor_cpu.sizes()[0] * tensor_cpu.sizes()[1]);
        for (size_t i = 0; i < last_elements.size(); ++i) {
            CHECK(doctest::Approx(data_ptr[i]) == static_cast<double>(last_elements[i]));
        }
    }
    SUBCASE((test_case_name + " - DeviceMatrices").c_str()) {
        DeviceMatrices<T> dm(4, 8, 16, 16, 16, 16, 16, queue);
        dm.fill(static_cast<T>(2.56f)).wait();
        torch::Tensor tensor = tnn::Module::convertDeviceMatricesToTorchTensor(dm);
        CHECK(tensor.device().is_xpu());

        torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat64);
        CHECK(tensor_cpu.device().is_cpu());
        auto result_vector = dm.copy_to_host();
        auto data_ptr = tensor_cpu.data_ptr<double>();

        for (size_t i = 0; i < result_vector.size(); ++i) {
            CHECK(doctest::Approx(data_ptr[i]) == static_cast<double>(result_vector[i]));
        }
    }

    SUBCASE((test_case_name + " - DeviceMem").c_str()) {
        DeviceMem<T> dm(16, queue);
        dm.fill(static_cast<T>(1.234)).wait();
        torch::Tensor tensor = tnn::Module::convertDeviceMemToTorchTensor(dm);
        CHECK(tensor.device().is_xpu());

        torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat64);
        CHECK(tensor_cpu.device().is_cpu());
        auto result_vector = dm.copy_to_host();
        auto data_ptr = tensor_cpu.data_ptr<double>();

        for (size_t i = 0; i < result_vector.size(); ++i) {
            CHECK(doctest::Approx(data_ptr[i]) == static_cast<double>(result_vector[i]));
        }
    }
}

TEST_CASE("DeviceMemory Tests") {
    sycl::queue queue; // Replace with actual queue initialization.

    // // Test with float
    FillAndTestDeviceMemory<float>("Float", queue);

    // Test with bfloat16
    FillAndTestDeviceMemory<bf16>("BFloat16", queue);

    // Test with half
    FillAndTestDeviceMemory<fp16>("Sycl half", queue);
}

TEST_CASE("Convert torch::Tensor to std::vector for different data types and devices") {
    double eps = 1e-2;
    // Tensor on CPU
    torch::Tensor cpu_tensor_int = torch::tensor({1, 2, 3, 4}, torch::kInt);
    torch::Tensor cpu_tensor_float = torch::tensor({1.1, 2.2, 3.3, 4.4}, torch::kFloat32);
    torch::Tensor cpu_tensor_double = torch::tensor({1.11, 2.22, 3.33, 4.44}, torch::kDouble);
    torch::Tensor cpu_tensor_bfloat16 = torch::tensor({11.111, 22.222, 33.333, 44.444}, torch::kBFloat16);
    torch::Tensor cpu_tensor_float16 = torch::tensor({111.1111, 222.2222, 333.3333, 444.4444}, torch::kFloat16);

    // Tensor on hypothetical XPU
    torch::Tensor xpu_tensor_int = cpu_tensor_int.to(torch::kXPU);
    torch::Tensor xpu_tensor_float = cpu_tensor_float.to(torch::kXPU);
    torch::Tensor xpu_tensor_double = cpu_tensor_double.to(torch::kXPU);
    torch::Tensor xpu_tensor_bfloat16 = cpu_tensor_bfloat16.to(torch::kXPU);
    torch::Tensor xpu_tensor_float16 = cpu_tensor_float16.to(torch::kXPU);

    SUBCASE("Convert int Tensor on CPU") {
        std::vector<int> v = tnn::Module::convertTensorToVector<int>(cpu_tensor_int);

        CHECK(v.size() == cpu_tensor_int.numel());
        CHECK(v[0] == 1);
        CHECK(v[1] == 2);
        CHECK(v[2] == 3);
        CHECK(v[3] == 4);
    }

    SUBCASE("Convert float Tensor on CPU") {
        std::vector<float> v = tnn::Module::convertTensorToVector<float>(cpu_tensor_float);

        CHECK(v.size() == cpu_tensor_float.numel());
        CHECK(v[0] == doctest::Approx(1.1f).epsilon(eps));
        CHECK(v[1] == doctest::Approx(2.2f).epsilon(eps));
        CHECK(v[2] == doctest::Approx(3.3f).epsilon(eps));
        CHECK(v[3] == doctest::Approx(4.4f).epsilon(eps));
    }

    SUBCASE("Convert double Tensor on CPU") {
        std::vector<double> v = tnn::Module::convertTensorToVector<double>(cpu_tensor_double);

        CHECK(v.size() == cpu_tensor_double.numel());
        CHECK(v[0] == doctest::Approx(1.11).epsilon(eps));
        CHECK(v[1] == doctest::Approx(2.22).epsilon(eps));
        CHECK(v[2] == doctest::Approx(3.33).epsilon(eps));
        CHECK(v[3] == doctest::Approx(4.44).epsilon(eps));
    }

    SUBCASE("Convert bf16 Tensor on CPU") {
        std::vector<bf16> v = tnn::Module::convertTensorToVector<bf16>(cpu_tensor_bfloat16);

        CHECK(v.size() == cpu_tensor_bfloat16.numel());
        CHECK(static_cast<double>(v[0]) == doctest::Approx(11.111).epsilon(eps));
        CHECK(static_cast<double>(v[1]) == doctest::Approx(22.222).epsilon(eps));
        CHECK(static_cast<double>(v[2]) == doctest::Approx(33.333).epsilon(eps));
        CHECK(static_cast<double>(v[3]) == doctest::Approx(44.444).epsilon(eps));
    }

    SUBCASE("Convert half Tensor on CPU") {
        std::vector<fp16> v = tnn::Module::convertTensorToVector<fp16>(cpu_tensor_float16);

        CHECK(v.size() == cpu_tensor_bfloat16.numel());
        CHECK(static_cast<double>(v[0]) == doctest::Approx(111.1111).epsilon(eps));
        CHECK(static_cast<double>(v[1]) == doctest::Approx(222.2222).epsilon(eps));
        CHECK(static_cast<double>(v[2]) == doctest::Approx(333.3333).epsilon(eps));
        CHECK(static_cast<double>(v[3]) == doctest::Approx(444.4444).epsilon(eps));
    }

    SUBCASE("Convert int Tensor on XPU") {
        std::vector<int> v = tnn::Module::convertTensorToVector<int>(xpu_tensor_int);

        CHECK(v.size() == xpu_tensor_int.numel());
        CHECK(v[0] == 1);
        CHECK(v[1] == 2);
        CHECK(v[2] == 3);
        CHECK(v[3] == 4);
    }

    SUBCASE("Convert float Tensor on XPU") {
        std::vector<float> v = tnn::Module::convertTensorToVector<float>(xpu_tensor_float);

        CHECK(v.size() == xpu_tensor_float.numel());
        CHECK(v[0] == doctest::Approx(1.1f).epsilon(eps));
        CHECK(v[1] == doctest::Approx(2.2f).epsilon(eps));
        CHECK(v[2] == doctest::Approx(3.3f).epsilon(eps));
        CHECK(v[3] == doctest::Approx(4.4f).epsilon(eps));
    }

    SUBCASE("Convert double Tensor on XPU") {
        std::vector<double> v = tnn::Module::convertTensorToVector<double>(xpu_tensor_double);

        CHECK(v.size() == xpu_tensor_double.numel());
        CHECK(v[0] == doctest::Approx(1.11).epsilon(eps));
        CHECK(v[1] == doctest::Approx(2.22).epsilon(eps));
        CHECK(v[2] == doctest::Approx(3.33).epsilon(eps));
        CHECK(v[3] == doctest::Approx(4.44).epsilon(eps));
    }

    SUBCASE("Convert bfloat16 Tensor on XPU") {
        std::vector<bf16> v = tnn::Module::convertTensorToVector<bf16>(xpu_tensor_bfloat16);

        CHECK(v.size() == xpu_tensor_bfloat16.numel());
        CHECK(static_cast<double>(v[0]) == doctest::Approx(11.111).epsilon(eps));
        CHECK(static_cast<double>(v[1]) == doctest::Approx(22.222).epsilon(eps));
        CHECK(static_cast<double>(v[2]) == doctest::Approx(33.333).epsilon(eps));
        CHECK(static_cast<double>(v[3]) == doctest::Approx(44.444).epsilon(eps));
    }

    SUBCASE("Convert half Tensor on XPU") {
        std::vector<fp16> v = tnn::Module::convertTensorToVector<fp16>(xpu_tensor_float16);

        CHECK(v.size() == xpu_tensor_float16.numel());
        CHECK(static_cast<double>(v[0]) == doctest::Approx(111.1111).epsilon(eps));
        CHECK(static_cast<double>(v[1]) == doctest::Approx(222.2222).epsilon(eps));
        CHECK(static_cast<double>(v[2]) == doctest::Approx(333.3333).epsilon(eps));
        CHECK(static_cast<double>(v[3]) == doctest::Approx(444.4444).epsilon(eps));
    }
}

TEST_CASE("Convert std::vector to torch::Tensor for different data types and devices") {
    double eps = 1e-2;
    // Vectors on CPU
    std::vector<int> cpu_vector_int = {1, 2, 3, 4};
    std::vector<float> cpu_vector_float = {1.1f, 2.2f, 3.3f, 4.4f};
    std::vector<double> cpu_vector_double = {1.11, 2.22, 3.33, 4.44};
    std::vector<bf16> cpu_vector_bfloat16 = {bf16(11.111f), bf16(22.222f), bf16(33.333f), bf16(44.444f)};
    std::vector<fp16> cpu_vector_float16 = {fp16(111.1111f), fp16(222.2222f), fp16(333.333f), fp16(444.4444f)};

    SUBCASE("Convert int vector to Tensor on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<int>(cpu_vector_int);

        CHECK(t.numel() == cpu_vector_int.size());
        CHECK(t[0].item<int>() == 1);
        CHECK(t[1].item<int>() == 2);
        CHECK(t[2].item<int>() == 3);
        CHECK(t[3].item<int>() == 4);
    }

    SUBCASE("Convert float vector to Tensor on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<float>(cpu_vector_float);

        CHECK(t.numel() == cpu_vector_float.size());
        CHECK(t[0].item<float>() == doctest::Approx(1.1f).epsilon(eps));
        CHECK(t[1].item<float>() == doctest::Approx(2.2f).epsilon(eps));
        CHECK(t[2].item<float>() == doctest::Approx(3.3f).epsilon(eps));
        CHECK(t[3].item<float>() == doctest::Approx(4.4f).epsilon(eps));
    }

    SUBCASE("Convert double vector to Tensor on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<double>(cpu_vector_double);

        CHECK(t.numel() == cpu_vector_double.size());
        CHECK(t[0].item<double>() == doctest::Approx(1.11).epsilon(eps));
        CHECK(t[1].item<double>() == doctest::Approx(2.22).epsilon(eps));
        CHECK(t[2].item<double>() == doctest::Approx(3.33).epsilon(eps));
        CHECK(t[3].item<double>() == doctest::Approx(4.44).epsilon(eps));
    }

    SUBCASE("Convert bf16 vector to Tensor on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<bf16>(cpu_vector_bfloat16);

        CHECK(t.numel() == cpu_vector_bfloat16.size());
        CHECK(static_cast<double>(t[0].item<float>()) == doctest::Approx(11.111).epsilon(eps));
        CHECK(static_cast<double>(t[1].item<float>()) == doctest::Approx(22.222).epsilon(eps));
        CHECK(static_cast<double>(t[2].item<float>()) == doctest::Approx(33.333).epsilon(eps));
        CHECK(static_cast<double>(t[3].item<float>()) == doctest::Approx(44.444).epsilon(eps));
    }
    SUBCASE("Convert fp16 vector to Tensor on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<fp16>(cpu_vector_float16);

        CHECK(t.numel() == cpu_vector_float16.size());
        CHECK(static_cast<double>(t[0].item<float>()) == doctest::Approx(111.1111).epsilon(eps));
        CHECK(static_cast<double>(t[1].item<float>()) == doctest::Approx(222.2222).epsilon(eps));
        CHECK(static_cast<double>(t[2].item<float>()) == doctest::Approx(333.3333).epsilon(eps));
        CHECK(static_cast<double>(t[3].item<float>()) == doctest::Approx(444.4444).epsilon(eps));
    }
}
TEST_CASE("Large Scale Vector to Tensor and Tensor to Vector Tests") {
    double eps = 1e-2;
    size_t large_size = 1e6;
    double value = 1.234;

    // Create large vectors
    std::vector<int> large_vector_int(large_size, static_cast<int>(value));
    std::vector<float> large_vector_float(large_size, static_cast<float>(value));
    std::vector<double> large_vector_double(large_size, value);
    std::vector<bf16> large_vector_bfloat16(large_size, bf16(static_cast<float>(value)));
    std::vector<fp16> large_vector_float16(large_size, fp16(static_cast<float>(value)));

    // Convert vectors to tensors and back
    SUBCASE("Convert large int vector to Tensor and back on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<int>(large_vector_int);
        std::vector<int> v = tnn::Module::convertTensorToVector<int>(t);

        CHECK(t.numel() == large_vector_int.size());
        CHECK(v.size() == large_vector_int.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(v[i] == large_vector_int[i]);
        }
    }

    SUBCASE("Convert large float vector to Tensor and back on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<float>(large_vector_float);
        std::vector<float> v = tnn::Module::convertTensorToVector<float>(t);

        CHECK(t.numel() == large_vector_float.size());
        CHECK(v.size() == large_vector_float.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(v[i] == doctest::Approx(large_vector_float[i]).epsilon(eps));
        }
    }

    SUBCASE("Convert large double vector to Tensor and back on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<double>(large_vector_double);
        std::vector<double> v = tnn::Module::convertTensorToVector<double>(t);

        CHECK(t.numel() == large_vector_double.size());
        CHECK(v.size() == large_vector_double.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(v[i] == doctest::Approx(large_vector_double[i]).epsilon(eps));
        }
    }

    SUBCASE("Convert large bf16 vector to Tensor and back on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<bf16>(large_vector_bfloat16);
        std::vector<bf16> v = tnn::Module::convertTensorToVector<bf16>(t);

        CHECK(t.numel() == large_vector_bfloat16.size());
        CHECK(v.size() == large_vector_bfloat16.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(static_cast<double>(v[i]) ==
                  doctest::Approx(static_cast<double>(large_vector_bfloat16[i])).epsilon(eps));
        }
    }

    SUBCASE("Convert large fp16 vector to Tensor and back on CPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<fp16>(large_vector_float16);
        std::vector<fp16> v = tnn::Module::convertTensorToVector<fp16>(t);

        CHECK(t.numel() == large_vector_float16.size());
        CHECK(v.size() == large_vector_float16.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(static_cast<double>(v[i]) ==
                  doctest::Approx(static_cast<double>(large_vector_float16[i])).epsilon(eps));
        }
    }

    // Convert vectors to tensors and back on XPU
    SUBCASE("Convert large int vector to Tensor and back on XPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<int>(large_vector_int).to(torch::kXPU);
        std::vector<int> v = tnn::Module::convertTensorToVector<int>(t.to(torch::kCPU));

        CHECK(t.numel() == large_vector_int.size());
        CHECK(v.size() == large_vector_int.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(v[i] == large_vector_int[i]);
        }
    }

    SUBCASE("Convert large float vector to Tensor and back on XPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<float>(large_vector_float).to(torch::kXPU);
        std::vector<float> v = tnn::Module::convertTensorToVector<float>(t.to(torch::kCPU));

        CHECK(t.numel() == large_vector_float.size());
        CHECK(v.size() == large_vector_float.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(v[i] == doctest::Approx(large_vector_float[i]).epsilon(eps));
        }
    }

    SUBCASE("Convert large double vector to Tensor and back on XPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<double>(large_vector_double).to(torch::kXPU);
        std::vector<double> v = tnn::Module::convertTensorToVector<double>(t.to(torch::kCPU));

        CHECK(t.numel() == large_vector_double.size());
        CHECK(v.size() == large_vector_double.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(v[i] == doctest::Approx(large_vector_double[i]).epsilon(eps));
        }
    }

    SUBCASE("Convert large bf16 vector to Tensor and back on XPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<bf16>(large_vector_bfloat16).to(torch::kXPU);
        std::vector<bf16> v = tnn::Module::convertTensorToVector<bf16>(t.to(torch::kCPU));

        CHECK(t.numel() == large_vector_bfloat16.size());
        CHECK(v.size() == large_vector_bfloat16.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(static_cast<double>(v[i]) ==
                  doctest::Approx(static_cast<double>(large_vector_bfloat16[i])).epsilon(eps));
        }
    }
    SUBCASE("Convert large fp16 vector to Tensor and back on XPU") {
        torch::Tensor t = tnn::Module::convertVectorToTensor<fp16>(large_vector_float16).to(torch::kXPU);
        std::vector<fp16> v = tnn::Module::convertTensorToVector<fp16>(t.to(torch::kCPU));

        CHECK(t.numel() == large_vector_float16.size());
        CHECK(v.size() == large_vector_float16.size());
        for (size_t i = 0; i < v.size(); ++i) {
            CHECK(static_cast<double>(v[i]) ==
                  doctest::Approx(static_cast<double>(large_vector_float16[i])).epsilon(eps));
        }
    }
}

TEST_CASE("Convert torch::Tensor to DeviceMatrix") {
    sycl::queue queue; // Assuming the queue has been initialized properly

    int batch_size = 8;
    int input_width = 16;
    float input_val = 1.0f;

    SUBCASE("Matching sizes should not throw") {
        torch::Tensor tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kXPU));
        DeviceMatrix<float> device_matrix(2, 3, queue);
        CHECK_NOTHROW(tnn::Module::convertTensorToDeviceMatrix(tensor, device_matrix, queue));
    }

    SUBCASE("Mismatching sizes should throw") {
        torch::Tensor tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kXPU));
        DeviceMatrix<float> device_matrix(4, 5, queue);

        CHECK_THROWS(tnn::Module::convertTensorToDeviceMatrix(tensor, device_matrix, queue));
    }

    SUBCASE("Convert float Tensor using tnn::Module::convertTensorToDeviceMatrix") {
        // Create an XPU tensor of type float
        torch::Tensor xpu_tensor_float =
            torch::ones({batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::Float) * input_val;
        DeviceMatrix<float> device_matrix(batch_size, input_width, queue);
        device_matrix.fill(0.0f).wait();
        // Convert Tensor to DeviceMatrix
        tnn::Module::convertTensorToDeviceMatrix<float>(xpu_tensor_float, device_matrix, queue);
        CHECK(device_matrix.rows() == batch_size);
        CHECK(device_matrix.cols() == input_width);

        // Check if the data has been copied correctly
        auto result_vector = device_matrix.copy_to_host();
        for (size_t i = 0; i < device_matrix.size(); ++i) {
            CHECK(result_vector[i] == input_val);
        }
    }

    SUBCASE("Convert half Tensor using tnn::Module::convertTensorToDeviceMatrix") {
        // Create an XPU tensor of type float
        torch::Tensor xpu_tensor_float =
            torch::ones({batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::Half) * input_val;
        DeviceMatrix<fp16> device_matrix(batch_size, input_width, queue);
        device_matrix.fill(0.0f).wait();
        // Convert Tensor to DeviceMatrix
        tnn::Module::convertTensorToDeviceMatrix<fp16>(xpu_tensor_float, device_matrix, queue);
        CHECK(device_matrix.rows() == batch_size);
        CHECK(device_matrix.cols() == input_width);
        // Check if the data has been copied correctly
        auto result_vector = device_matrix.copy_to_host();
        for (size_t i = 0; i < device_matrix.size(); ++i) {
            CHECK(result_vector[i] == input_val);
        }
    }
    SUBCASE("Convert bf16 Tensor using tnn::Module::convertTensorToDeviceMatrix") {
        // Create an XPU tensor of type float
        torch::Tensor xpu_tensor_float =
            torch::ones({batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16) * input_val;
        DeviceMatrix<bf16> device_matrix(batch_size, input_width, queue);
        device_matrix.fill(0.0f).wait();

        // Convert Tensor to DeviceMatrix
        tnn::Module::convertTensorToDeviceMatrix<bf16>(xpu_tensor_float, device_matrix, queue);
        CHECK(device_matrix.rows() == batch_size);
        CHECK(device_matrix.cols() == input_width);

        // Check if the data has been copied correctly
        auto result_vector = device_matrix.copy_to_host();
        for (size_t i = 0; i < device_matrix.size(); ++i) {
            CHECK(result_vector[i] == input_val);
        }
    }
}