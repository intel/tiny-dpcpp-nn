/**
 * @file result_check.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Basic comparison, load, store functionalities to check correctness of results.
 * TODO: move the load/store functionalities in a different file and put everything in a namespace.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

// Uncomment the following line to enable printing
#define ENABLE_PRINTING
template <typename T> double GetInfNorm(const std::vector<T> &v) {
    double norm = 0.0;
    for (auto val : v) {
        norm = std::max(norm, std::abs((double)val));
    }

    return norm;
}

template <typename Tl, typename Tr>
std::vector<double> GetAbsDiff(const std::vector<Tl> &lhs, const std::vector<Tr> &rhs) {

    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Size mismatch: lhs size = " + std::to_string(lhs.size()) +
                                    ", rhs size = " + std::to_string(rhs.size()));
    }
    std::vector<double> ret(lhs.size(), 0.0);

    for (size_t iter = 0; iter < lhs.size(); iter++) {
        if (!std::isfinite(lhs[iter]) || !std::isfinite(rhs[iter])) throw std::invalid_argument("Infinite numbers");
        ret[iter] = std::abs(lhs[iter] - rhs[iter]);
    }

    return ret;
}

template <typename Tl, typename Tr> std::vector<double> GetAbsDiff(const std::vector<Tl> &lhs, const Tr rhs) {
    std::vector<double> ret(lhs.size(), 0.0);

    for (size_t iter = 0; iter < lhs.size(); iter++) {
        if (!std::isfinite(lhs[iter]) || !std::isfinite(rhs)) throw std::invalid_argument("Infinite numbers");
        ret[iter] = std::abs(lhs[iter] - rhs);
    }

    return ret;
}

template <typename Tval, typename Ttarget>
bool isVectorWithinTolerance(const std::vector<Tval> &value, const Ttarget target, const double tolerance) {

    bool is_same = true;
    double max_diff = 0.0;
    const double inf_diff = GetInfNorm(GetAbsDiff(value, target));
    const double inf_val = GetInfNorm(value);
    if ((double)target == 0.0)
        max_diff = inf_diff;
    else
        max_diff = inf_diff / std::max(std::abs((double)target), inf_val);

    if (max_diff > tolerance) is_same = false;
    if (!is_same) {
        std::cout << "Values are within tolerance = " << std::boolalpha << is_same << std::noboolalpha
                  << ". Max diff = " << max_diff << std::endl;
    }

    return is_same;
}

template <typename Tval, typename Ttarget>
bool areVectorsWithinTolerance(const std::vector<Tval> &value, const std::vector<Ttarget> &target,
                               const double tolerance) {

    bool is_same = true;
    double max_diff = 0.0;
    const double inf_diff = GetInfNorm(GetAbsDiff(value, target));
    const double inf_val = GetInfNorm(value);
    const double inf_tar = GetInfNorm(target);
    if ((double)inf_tar == 0.0)
        max_diff = inf_diff;
    else
        max_diff = inf_diff / std::max(inf_tar, inf_val);

    if (max_diff > tolerance) is_same = false;
    if (!is_same) {
        std::cout << "Values are within tolerance = " << std::boolalpha << is_same << std::noboolalpha
                  << ". Max diff = " << max_diff << std::endl;
    }
    return is_same;
}

template <typename Tval, typename Ttarget>
bool areScalarsWithinTolerance(const Tval &value, const Ttarget &target, const double tolerance) {

    bool is_same = true;
    double max_diff = 0.0;
    const double inf_diff = std::abs(value - target);
    const double inf_val = std::abs(value);
    const double inf_tar = std::abs(target);
    if ((double)inf_tar == 0.0)
        max_diff = inf_diff;
    else
        max_diff = inf_diff / std::max(inf_tar, inf_val);

    if (max_diff > tolerance) is_same = false;

    if (!is_same) {
        std::cout << "Values are within tolerance = " << std::boolalpha << is_same << std::noboolalpha
                  << ". Max diff = " << max_diff << std::endl;
    }
    return is_same;
}

template <typename Iterator> void printElements(Iterator begin, Iterator end, const std::string &delimiter = ", ") {
    for (Iterator it = begin; it != end; ++it) {
        std::cout << *it << (std::next(it) != end ? delimiter : "");
    }
}

template <typename T>
void printVector(const std::string &name, const std::vector<T> &vec, int break_every = 0, int cutoff_val = 128) {
#ifdef ENABLE_PRINTING
    std::cout << "================================" << name << "============================" << std::endl;
    const size_t vec_size = vec.size();
    int counter = 0;

    bool print_all = cutoff_val <= 0 || vec_size <= cutoff_val;

    if (!print_all) {
        // Print first half
        printElements(vec.begin(), vec.begin() + cutoff_val / 2);

        // Print ellipsis
        std::cout << "..., ";

        // Print last half
        printElements(vec.end() - cutoff_val / 2, vec.end());
    } else {
        // Print all elements with optional line breaks
        for (auto it = vec.begin(); it != vec.end(); ++it) {
            if (break_every && counter >= break_every) {
                std::cout << std::endl;
                if (break_every < vec_size) { // Only print the separator if break_every is used
                    std::cout << "------------------------------------------------------------" << std::endl;
                }
                counter = 0;
            }
            // Print current element
            std::cout << *it << (std::next(it) != vec.end() ? ", " : "");
            counter++;
        }
    }
    std::cout << std::endl;
#endif
}