/**
 * @file DeviceMem.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implements a class which holds device memory. Not really required anymore and should be removed in the future
 * in favor of DeviceMatrix.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <atomic>
#include <dpct/dpct.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

#include "common.h"

using namespace sycl;

/**
 * @brief Class which manages device memory. Cannot be used on the device.
 *
 * TODO: remove in favor of DeviceMatrix/DeviceMatrices. Consolidate with DeviceMatrix classes.
 *
 * @tparam T
 */
template <typename T> class DeviceMem {
  private:
    size_t m_size;
    sycl::queue &m_q;
    T *m_data = nullptr;

  public:
    // Default constructor
    DeviceMem() = delete;

    /**
     * @brief Constructor for the DeviceMem class
     *
     * @param size               Size of the memory to allocate in elements.
     * @param queue              SYCL queue associated with the object.
     */
    DeviceMem(const size_t size, sycl::queue &q) : m_size(size), m_q(q) {
        // allocate less than 4 GB in one go.
        /// TODO: make sure this works with arbitrary sizes and that we only take the device memory as limit
        // if ((size() * sizeof(T)) > ((size_t)4 * (1 << 30)))
        //     throw std::invalid_argument("Trying to allocate too large DeviceMem (> 4GB).");
        if (size <= 0) throw std::invalid_argument("0 size is not allowed.");
        m_data = sycl::malloc_device<T>(m_size, m_q);
    }

    ~DeviceMem() { sycl::free(m_data, m_q); }

    // Copy data from host to device
    sycl::event copy_from_host(const std::vector<T> &data) {
        assert(data.size() == size());
        return m_q.memcpy(m_data, data.data(), get_bytes());
    }

    // Copy data from device to host
    sycl::event copy_to_host(std::vector<T> &data) const {
        assert(data.size() == size());
        return m_q.memcpy(data.data(), m_data, get_bytes());
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> ret(size());
        copy_to_host(ret).wait();
        return ret;
    }

    /// Copies size elements from another device array to this one, automatically
    /// resizing it
    sycl::event copy_from_device(const DeviceMem<T> &other) {
        assert(other.size() <= size());

        return m_q.memcpy(m_data, other.m_data, other.get_bytes());
    }

    /// copy and cast from another device array.
    /// Note that the size of the src array has to be equal or larger than the size of the target array
    /// TODO: get rid of this.
    template <typename Tsrc>
    static sycl::event copy_from_device(DeviceMem<T> &target, Tsrc const *const src, sycl::queue &q) {
        T *const ptr = target.data();
        return q.parallel_for(target.size(), [=](auto idx) { ptr[idx] = static_cast<T>(src[idx]); });
    }

    // Get the raw data pointer
    T const *const data() const { return m_data; }
    T *const data() { return m_data; }

    /// Sets the memory of the all elements to value
    sycl::event fill(const T &value) { return m_q.fill(m_data, value, size()); }

    // Get the size of the memory allocation
    size_t size() const { return m_size; }

    // Get bytes of allocated memory size
    size_t get_bytes() const { return size() * sizeof(T); }

    static void save_to_file(const DeviceMem<T> &vec, std::string filename) {
        // Open the file for writing
        std::ofstream file;
        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        std::vector<T> host(vec.size());
        vec.copy_to_host(host).wait();

        // Write each value of the weights matrices to the file
        for (int i = 0; i < host.size(); i++) {
            file << (float)host[i] << "\n";
        }

        // Close the file
        file.close();
    }
};