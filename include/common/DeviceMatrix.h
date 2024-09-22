/**
 * @file DeviceMatrix.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementations of host and device representations of a single matrix
 * and multiple matrices.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "common.h"

#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

enum class MatrixLayout { RowMajor = 0, ColumnMajor = 1 };

/**
 * @brief View class for a single matrix which does not own any memory.
 * This can be used on the device if the pointer ptr is a device pointer.
 *
 * @tparam T
 */
template <typename T> class DeviceMatrixView {
  public:
    DeviceMatrixView() = delete;
    DeviceMatrixView(const size_t m, const size_t n, const size_t stride_col, T *ptr)
        : m_(m), n_(n), stride_col_(stride_col), ptr_(ptr) {}

    // T &operator()(const int i, const int j) { return ptr_[j + i * stride_col_]; }
    T &operator()(const size_t i, const size_t j) const { return ptr_[j + i * stride_col_]; }

    T *const GetPointer(const size_t i, const size_t j) const { return ptr_ + j + i * stride_col_; }
    T *const GetPointer() const { return ptr_; }

    DeviceMatrixView<T> GetSubMatrix(const size_t m, const size_t n, const size_t offset_m,
                                     const size_t offset_n) const {
        return DeviceMatrixView<T>(m, n, stride_col_, ptr_ + offset_n + offset_m * stride_col_);
    }

    size_t m() const { return m_; }
    size_t n() const { return n_; }
    size_t nelements() const { return m() * n(); }

  private:
    const size_t m_;
    const size_t n_;
    const size_t stride_col_;
    T *const ptr_;
};

/**
 * @brief A view class which can be used on the device and represents multiple
 * matrices. One input matrix, one output matrix and n_matrices-2 middle matrices,
 * Each of the three types can have different dimensions.
 *
 * @tparam T
 */
template <typename T> class DeviceMatricesView {
  public:
    DeviceMatricesView() = delete;
    DeviceMatricesView(const uint32_t n_matrices, const size_t input_m, const size_t input_n, const size_t middle_m,
                       const size_t middle_n, const size_t output_m, const size_t output_n, T *ptr)
        : n_matrices_(n_matrices), input_m_(input_m), input_n_(input_n), middle_m_(middle_m), middle_n_(middle_n),
          output_m_(output_m), output_n_(output_n), ptr_(ptr) {}

    T *const GetMatrixPointer(const uint32_t matrix) const {
        if (matrix == 0)
            return ptr_;
        else if (matrix < n_matrices_)
            return ptr_ + input_m_ * input_n_ + (matrix - 1) * middle_m_ * middle_n_;
        return nullptr;
    }

    /// Get the row,col element in the given matrix. Undefined behavior if the
    /// parameters point to a memory location not available in this
    T *const GetElementPointer(const uint32_t matrix, const size_t row, const size_t col) const {
        T *const mat_ptr = GetMatrixPointer(matrix);
        if (matrix == 0)
            return mat_ptr + row * input_n_ + col;
        else if (matrix == n_matrices_ - 1)
            return mat_ptr + row * output_n_ + col;
        else
            return mat_ptr + row * middle_n_ + col;
    }

    size_t nelements() const {
        return input_m_ * input_n_ + output_m_ * output_n_ + (n_matrices_ - 2) * middle_m_ * middle_n_;
    }

  private:
    const uint32_t n_matrices_;
    const size_t input_m_;
    const size_t input_n_;
    const size_t middle_m_;
    const size_t middle_n_;
    const size_t output_m_;
    const size_t output_n_;
    T *const ptr_;
};

/**
 * @brief Class which represents a matrix on the device.
 * This class owns memory and the queue. It cannot be used directly on the device
 * but can only be accessed on the device through a DeviceMatrixView or DeviceMatricesView
 * class.
 *
 * @tparam T
 * @tparam _layout
 */
template <typename T, MatrixLayout _layout = MatrixLayout::RowMajor> class DeviceMatrix {
  public:
    // Owning its memory as an allocation from a stream's memory arena
    DeviceMatrix(const size_t m, const size_t n, sycl::queue &stream)
        : m_rows(m), m_cols(n), m_q(stream), m_data(sycl::malloc_device<T>(m * n, stream)) {
        static_assert(_layout != MatrixLayout::ColumnMajor);
    }
    DeviceMatrix() = delete;

    DeviceMatrix(const DeviceMatrix<T, _layout> &other) : m_rows(other.m_rows), m_cols(other.m_cols), m_q(other.m_q) {
        m_data = sycl::malloc_device<T>(n_elements(), m_q);
        m_q.memcpy(m_data, other.m_data, n_bytes()).wait();
        static_assert(_layout != MatrixLayout::ColumnMajor);
    }
    DeviceMatrix(DeviceMatrix<T, _layout> &&other)
        : m_rows(other.m_rows), m_cols(other.m_cols), m_q(other.m_q), m_data(other.m_data) {
        other.m_data = nullptr;
        static_assert(_layout != MatrixLayout::ColumnMajor);
    }

    virtual ~DeviceMatrix() { sycl::free(m_data, m_q); }

    DeviceMatrix<T, _layout> &operator=(const DeviceMatrix<T, _layout> &other) {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Cannot assign matrices of differing dimensions.");
        if (m_q != other.m_q) throw std::invalid_argument("Cannot assign matrices with differing queues.");

        m_q.memcpy(m_data, other.m_data, n_bytes()).wait();
        return *this;
    }

    bool operator==(const DeviceMatrix<T, _layout> &rhs) const {
        if (this->rows() != rhs.rows() || this->cols() != rhs.cols()) return false;
        if (this->layout() != rhs.layout()) return false;

        // Check actual data
        std::vector<T> data1 = this->copy_to_host();
        std::vector<T> data2 = rhs.copy_to_host();
        for (size_t i = 0; i < data1.size(); ++i) {
            if (data1[i] != data2[i]) {
                return false;
            }
        }

        return true;
    }

    sycl::event fill(const T val) { return m_q.fill(data(), val, n_elements()); }

    void fillSubmatrixWithValue(size_t start_row, size_t start_col, size_t submatrix_rows, size_t submatrix_cols,
                                T fill_value) {
        // Error checking to make sure submatrix is within bounds
        if (start_row + submatrix_rows > m_rows || start_col + submatrix_cols > m_cols) {
            throw std::invalid_argument("Submatrix dimensions exceed matrix bounds.");
        }

        T *data_ptr = this->data();
        size_t global_width = this->cols();

        // Use a kernel to fill the submatrix with the specified value
        m_q.parallel_for(sycl::range<1>{submatrix_rows * submatrix_cols},
                         [=](sycl::id<1> idx) {
                             size_t i = idx[0] / submatrix_cols; // Row index within submatrix
                             size_t j = idx[0] % submatrix_cols; // Column index within submatrix

                             // Calculate actual index in the overall matrix
                             size_t global_idx = (start_row + i) * global_width + (start_col + j);

                             // Assign the fill_value to the submatrix element
                             data_ptr[global_idx] = fill_value;
                         })
            .wait(); // Synchronize operations
    }

    void copyHostToSubmatrix(const std::vector<T> &srcHostVector, size_t start_row, size_t start_col, size_t rows,
                             size_t cols) {
        if (start_row + rows > m_rows || start_col + cols > m_cols) {
            throw std::invalid_argument("Submatrix dimensions exceed the bounds of the destination matrix.");
        }
        if (srcHostVector.size() < rows * cols) {
            throw std::invalid_argument("Source host vector size is too small for the specified submatrix dimensions.");
        }

        // Perform the copy in chunks corresponding to each row of the submatrix
        for (size_t row = 0; row < rows; ++row) {
            // Compute the offset into the source and destination data
            size_t srcOffset = row * cols;
            size_t dstOffset = (start_row + row) * m_cols + start_col;

            // Enqueue a copy operation for the current row
            m_q.memcpy(m_data + dstOffset, srcHostVector.data() + srcOffset, cols * sizeof(T)).wait();
        }
    }

    sycl::event copy_to_host(std::vector<T> &out) const {
        if (out.size() < n_elements()) throw std::invalid_argument("Target too small.");
        return m_q.memcpy(out.data(), data(), n_bytes());
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> v(n_elements());
        copy_to_host(v).wait();
        return v;
    }

    sycl::event copy_from_host(const std::vector<T> &vec) {
        if (vec.size() != n_elements())
            throw std::invalid_argument("Vector not same size as matrix. Input: " + std::to_string(vec.size()) +
                                        ", n_elements: " + std::to_string(n_elements()));
        return m_q.memcpy(data(), vec.data(), n_bytes());
    }

    template <typename Ts> void copy_from_device(Ts const *const src) {
        T *const ptr = m_data;
        m_q.parallel_for(size(), [=](auto idx) { ptr[idx] = static_cast<T>(src[idx]); }).wait();
    }

    T *data() { return m_data; }
    T const *const data() const { return m_data; }

    size_t rows() const { return m_rows; }
    size_t m() const { return rows(); }

    size_t cols() const { return m_cols; }
    size_t n() const { return cols(); }

    size_t n_elements() const { return rows() * cols(); }
    size_t size() const { return n_elements(); }
    size_t n_bytes() const { return n_elements() * sizeof(T); }

    // why? This just does not make any sense
    uint32_t stride() const { return _layout == MatrixLayout::ColumnMajor ? m() : n(); }

    constexpr MatrixLayout layout() const { return _layout; }

    DeviceMatrixView<T> GetView() { return GetView(m_rows, m_cols, 0, 0); }
    const DeviceMatrixView<T> GetView() const { return GetView(m_rows, m_cols, 0, 0); }

    // Get a matrices view for a single device matrix. Just a convenience function so we do not have
    // to manually shuffle data around.
    DeviceMatricesView<T> GetViews() const { return DeviceMatricesView<T>(1, m(), n(), 0, 0, 0, 0, m_data); }

    DeviceMatrixView<T> GetView(const size_t m, const size_t n, const size_t offset_m, const size_t offset_n) {
        if (offset_m + m > m_rows) throw std::invalid_argument("Potential OOB access.");
        if (offset_n + n > m_cols) throw std::invalid_argument("Potential OOB access.");

        return DeviceMatrixView<T>(m, n, m_cols, m_data + offset_n + offset_m * m_cols);
    }

    const DeviceMatrixView<T> GetView(const size_t m, const size_t n, const size_t offset_m,
                                      const size_t offset_n) const {
        if (offset_m + m > m_rows) throw std::invalid_argument("Potential OOB access.");
        if (offset_n + n > m_cols) throw std::invalid_argument("Potential OOB access.");

        return DeviceMatrixView<T>(m, n, m_cols, m_data + offset_n + offset_m * m_cols);
    }

    void print(int is_packed = 0) const {
        std::vector<T> data = copy_to_host();
        size_t num_rows_to_print = (this->rows() <= 10) ? this->rows() : 5;

        std::cout << "Matrix (" << this->rows() << "x" << this->cols() << "):" << std::endl;
        for (size_t i = 0; i < num_rows_to_print; ++i) {
            print_row(data, i, is_packed);
        }

        // Print ellipsis if rows are more than 10
        if (this->rows() > 10) {
            std::cout << "..." << std::endl;
            for (size_t i = this->rows() - num_rows_to_print; i < this->rows(); ++i) {
                print_row(data, i, is_packed);
            }
        }
    }

  private:
    void print_row(const std::vector<T> &data, size_t row_index, int is_packed) const {
        std::cout << "[ ";
        for (size_t j = 0; j < this->cols(); ++j) {
            size_t idx;
            if (_layout == MatrixLayout::ColumnMajor) {
                idx = j * rows() + row_index;
            } else {
                idx = row_index * cols() + j;
            }
            if (is_packed) {
                idx = toPackedLayoutCoord(idx, this->rows(), this->cols());
            }
            std::cout << static_cast<double>(data[idx]);
            if (j < this->cols() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]" << std::endl;
    }
    const size_t m_rows, m_cols;
    sycl::queue &m_q;
    T *m_data;
};

/**
 * @brief Class which represents multiple device matrices as required by our
 * MLP algorithms. This class owns the device memory and cannot be used on the device directly.
 * Usage on the device has to be through the DeviceMatrixView and DeviceMatricesView classes.
 * TODO: consolidate this class with DeviceMatrix class and DeviceMem class, they are all just
 * containers which hold and manage device memory.
 *
 * @tparam T
 */
template <typename T> class DeviceMatrices {

  public:
    DeviceMatrices() = delete;
    DeviceMatrices(const uint32_t n_matrices, const size_t input_m, const size_t input_n, const size_t middle_m,
                   const size_t middle_n, const size_t output_m, const size_t output_n, sycl::queue &q)
        : m_q(q), n_matrices_(n_matrices), input_m_(input_m), input_n_(input_n), middle_m_(middle_m),
          middle_n_(middle_n), output_m_(output_m), output_n_(output_n) {
        if (n_matrices_ < 2) throw std::invalid_argument("need to have at least 2 matrices.");
        matrices_ = sycl::malloc_device<T>(nelements(), m_q);
    }
    DeviceMatrices(const DeviceMatrices<T> &rhs) = delete;
    DeviceMatrices(DeviceMatrices<T> &&rhs) = delete;
    DeviceMatrices<T> &operator=(const DeviceMatrices<T> &rhs) = delete;
    DeviceMatrices<T> &operator=(DeviceMatrices<T> &&rhs) = delete;

    ~DeviceMatrices() { sycl::free(matrices_, m_q); }

    uint32_t GetNumberOfMatrices() const { return n_matrices_; }

    DeviceMatricesView<T> GetViews() const {
        return DeviceMatricesView<T>(n_matrices_, input_m_, input_n_, middle_m_, middle_n_, output_m_, output_n_,
                                     matrices_);
    }

    DeviceMatrixView<T> GetView(const uint32_t idx) const {
        size_t n = middle_n_;
        size_t m = middle_m_;
        if (idx == 0) {
            m = input_m_;
            n = input_n_;
        } else if (idx == n_matrices_ - 1) {
            m = output_m_;
            n = output_n_;
        }
        return DeviceMatrixView<T>(m, n, n, GetMatrixPtr(idx));
    }
    DeviceMatrixView<T> Front() const { return GetView(0); }
    DeviceMatrixView<T> Back() const { return GetView(n_matrices_ - 1); }

    void Transpose(DeviceMatrices<T> &ret) const {
        if (GetNumberOfMatrices() != ret.GetNumberOfMatrices())
            throw std::invalid_argument("Need to have same number of matrices for transpose");

        for (uint32_t iter = 0; iter < GetNumberOfMatrices(); iter++) {
            DeviceMatrices<T>::Transpose(GetView(iter), ret.GetView(iter), m_q);
        }
    }

    void PackAndTranspose(DeviceMatrices<T> &ret) const {
        if (GetNumberOfMatrices() != ret.GetNumberOfMatrices())
            throw std::invalid_argument("Need to have same number of matrices for transpose");

        for (uint32_t iter = 0; iter < GetNumberOfMatrices(); iter++) {
            DeviceMatrices<T>::PackAndTranspose(GetView(iter), ret.GetView(iter), m_q);
        }
    }

    void PackedTranspose(DeviceMatrices<T> &ret) const {
        if (GetNumberOfMatrices() != ret.GetNumberOfMatrices())
            throw std::invalid_argument("Need to have same number of matrices for transpose");

        for (uint32_t iter = 0; iter < GetNumberOfMatrices(); iter++) {
            DeviceMatrices<T>::PackedTranspose(GetView(iter), ret.GetView(iter), m_q);
        }
    }

    void Packed(DeviceMatrices<T> &ret) const {
        if (GetNumberOfMatrices() != ret.GetNumberOfMatrices())
            throw std::invalid_argument("Need to have same number of matrices for transpose");

        for (uint32_t iter = 0; iter < GetNumberOfMatrices(); iter++) {
            DeviceMatrices<T>::Packed(GetView(iter), ret.GetView(iter), m_q);
        }
    }

    sycl::event copy_from_host(const std::vector<T> &src) {
        return m_q.memcpy(matrices_, src.data(), nelements() * sizeof(T));
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> ret(nelements());
        m_q.memcpy(ret.data(), matrices_, nelements() * sizeof(T)).wait();
        return ret;
    }

    sycl::event fill(const T val) { return m_q.fill(matrices_, val, nelements()); }

    size_t nelements() const {
        return input_m_ * input_n_ + output_m_ * output_n_ + (n_matrices_ - 2) * middle_m_ * middle_n_;
    }

    size_t input_m() const { return input_m_; }
    size_t input_n() const { return input_n_; }
    size_t middle_m() const { return middle_m_; }
    size_t middle_n() const { return middle_n_; }
    size_t output_m() const { return output_m_; }
    size_t output_n() const { return output_n_; }

  private:
    T *GetMatrixPtr(const uint32_t matrix) const {
        if (matrix == 0)
            return matrices_;
        else if (matrix < n_matrices_)
            return matrices_ + input_m_ * input_n_ + (matrix - 1) * middle_m_ * middle_n_;
        else
            throw std::invalid_argument("matrix does not exist");

        return nullptr;
    }

    static void Transpose(const DeviceMatrixView<T> &src, DeviceMatrixView<T> dest, sycl::queue &q) {
        if (src.n() != dest.m() || src.m() != dest.n()) throw std::invalid_argument("Cannot transpose.");
        // TODO: check that the underlying data is actually in the same context.

        // Ensure that src and dest are not the same matrix
        if (src.GetPointer() == dest.GetPointer()) {
            throw std::invalid_argument("Cannot transpose in place: src and dest are the same.");
        }

        T *const new_p = dest.GetPointer();
        T const *const old_p = src.GetPointer();
        const size_t loc_cols = src.n();
        const size_t loc_rows = src.m();
        q.parallel_for(loc_rows * loc_cols, [=](auto idx) {
            const size_t row = idx / loc_cols;
            const size_t col = idx % loc_cols;
            const size_t new_idx = row + col * loc_rows;
            new_p[new_idx] = old_p[idx];
        });
    }

    // TODO, make this work in dependence of the data type.
    // Transposes the data assuming it is in a packed format
    static void PackedTranspose(const DeviceMatrixView<T> &src, DeviceMatrixView<T> dest, sycl::queue &q) {
        if (src.n() != dest.m() || src.m() != dest.n()) throw std::invalid_argument("Cannot transpose.");
        // TODO: check that the underlying data is actually in the same context.

        T *const transposed_p = dest.GetPointer();
        T const *const old_p = src.GetPointer();
        const size_t loc_rows = src.m();
        const size_t loc_cols = src.n();
        q.parallel_for(loc_rows * loc_cols, [=](auto idx) {
            const size_t i = idx / loc_cols;
            const size_t j = idx % loc_cols;
            int transposed_idx = j * loc_rows + i;
            transposed_p[toPackedLayoutCoord(idx, loc_cols, loc_rows)] =
                old_p[toPackedLayoutCoord(transposed_idx, loc_rows, loc_cols)];
        });
    }

    // TODO, make this work in dependence of the data type.
    // Packs an unpacked DeviceMatrixView src to a packed DeviceMatrixView
    static void Packed(const DeviceMatrixView<T> &src, DeviceMatrixView<T> dest, sycl::queue &q) {
        if (src.n() != dest.m() || src.m() != dest.n()) throw std::invalid_argument("Cannot transpose.");
        // TODO: check that the underlying data is actually in the same context.

        // Allocate temporary buffer for the source data
        // T const *const old_p = src.GetPointer();
        T *temp_src = sycl::malloc_device<T>(src.m() * src.n(), q);
        q.memcpy(temp_src, src.GetPointer(), src.m() * src.n() * sizeof(T)).wait();

        T *const new_p = dest.GetPointer();
        const size_t loc_rows = src.m();
        const size_t loc_cols = src.n();
        q.parallel_for(loc_rows * loc_cols, [=](auto idx) {
             new_p[toPackedLayoutCoord(idx, loc_cols, loc_rows)] = temp_src[idx];
         }).wait();

        // Free the temporary source buffer
        sycl::free(temp_src, q);
    }

    // Packs data and then transposes
    static void PackAndTranspose(const DeviceMatrixView<T> &src, DeviceMatrixView<T> dest, sycl::queue &q) {
        if (src.n() != dest.m() || src.m() != dest.n()) throw std::invalid_argument("Cannot transpose.");
        // TODO: check that the underlying data is actually in the same context.

        T *temp_src = sycl::malloc_device<T>(src.m() * src.n(), q);
        q.memcpy(temp_src, src.GetPointer(), src.m() * src.n() * sizeof(T)).wait();

        T *const transposed_p = dest.GetPointer();
        const size_t loc_rows = src.m();
        const size_t loc_cols = src.n();
        q.parallel_for(loc_rows * loc_cols, [=](auto idx) {
            const size_t i = idx / loc_cols;
            const size_t j = idx % loc_cols;
            int transposed_idx = j * loc_rows + i;
            transposed_p[toPackedLayoutCoord(idx, loc_rows, loc_cols)] = temp_src[transposed_idx];
        });

        // Free the temporary source buffer
        sycl::free(temp_src, q);
    }

    sycl::queue &m_q;
    const uint32_t n_matrices_;
    const size_t input_m_;
    const size_t input_n_;
    const size_t middle_m_;
    const size_t middle_n_;
    const size_t output_m_;
    const size_t output_n_;
    T *matrices_;
};
