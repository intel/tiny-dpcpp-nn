#pragma once

#include "doctest/doctest.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <exception>

#include "DeviceMatrix.h"
#include "common/common.h"
#include "mlp.h"

namespace test {
namespace common {

std::string GenerateTestName(const size_t batch_size, const size_t width, const Activation activation,
                             const Activation output_activation, const mlp_cpp::WeightInitMode weight_init_mode, 
                             const bool random_init) 
{
    std::string random_string = random_init ? "true" : "false";
    std::string testName =
                            "Testing loss WIDTH " + std::to_string(width) + 
                            " - activation: " + std::to_string((int)activation) +
                            " - output_activation: " + std::to_string((int)output_activation) + 
                            " - weight_init_mode: " + std::to_string((int)weight_init_mode) +
                            " - Batch size : " + std::to_string(batch_size) + 
                            " - random input:" + random_string;
                            
    return testName;
}

void LoopOverParams(sycl::queue& q, const std::vector<int>& batch_sizes, 
    const std::vector<int>& widths, 
    const std::vector<Activation>& activations,
    const std::vector<Activation>& output_activations,
    const std::vector<mlp_cpp::WeightInitMode>& weight_init_modes, 
    const std::vector<bool>& random_inputs,
    std::function<void(sycl::queue& q, const int, const int, const Activation,const Activation, const mlp_cpp::WeightInitMode, const bool)> test) {
    for (auto batch_size : batch_sizes) {
    for (auto width : widths) {
    for (auto activation : activations) {
    for (auto output_activation : output_activations) {
    for (auto weight_init_mode : weight_init_modes) {
    for (auto random_input : random_inputs) {
        SUBCASE(GenerateTestName(batch_size, width, activation, output_activation, weight_init_mode, random_input).c_str()) {
            CHECK_NOTHROW(test(q, batch_size, width, activation, output_activation, weight_init_mode, random_input););
        }
    }
    }
    }
    }
    }
    }
}

template <typename T>
void copyHostToSubmatrix(DeviceMatrix<T>& mat, const std::vector<T> &srcHostVector, size_t start_row, size_t start_col, size_t rows,
                            size_t cols, sycl::queue& Q) {
    if (start_row + rows > mat.m() || start_col + cols > mat.n()) {
        throw std::invalid_argument("Submatrix dimensions exceed the bounds of the destination matrix.");
    }
    if (srcHostVector.size() < rows * cols) {
        throw std::invalid_argument("Source host vector size is too small for the specified submatrix dimensions.");
    }

    // Perform the copy in chunks corresponding to each row of the submatrix
    for (size_t row = 0; row < rows; ++row) {
        // Compute the offset into the source and destination data
        size_t srcOffset = row * cols;
        size_t dstOffset = (start_row + row) * mat.n() + start_col;

        // Enqueue a copy operation for the current row
        Q.memcpy(mat.data() + dstOffset, srcHostVector.data() + srcOffset, cols * sizeof(T)).wait();
    }
}

///helper functions
template <typename T>
void fillSubmatrixWithValue(DeviceMatrix<T>& mat, size_t start_row, size_t start_col, size_t submatrix_rows, size_t submatrix_cols,
                            const T fill_value, sycl::queue& Q) {
    std::vector<T> fill_vals(submatrix_rows * submatrix_cols, fill_value);
    copyHostToSubmatrix(mat, fill_vals, start_row, start_col, submatrix_rows, submatrix_cols, Q);
}



template <typename T> std::vector<T> create_padded_vector(int output_width, T target_val, int padded_output_width) {
    if (output_width > padded_output_width) {
        throw std::invalid_argument("output_width cannot be greater than N");
    }

    // Initialize a vector of N zeros
    std::vector<T> target_ref(padded_output_width, 0);
    // Set the first output_width elements to target_val
    std::fill_n(target_ref.begin(), output_width, target_val);
    return target_ref;
}


}
}