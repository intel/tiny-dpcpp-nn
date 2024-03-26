/**
 * @file common.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Definitions of common funcitonalities which are not templated.
 * TODO: move all non-templated common functions here.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "common.h"
#include <algorithm> //std::equal
#include <cctype>    //std::tolower
#include <iostream>

/**
 * Convert an index to a packed layout coordinate for interleaved even and odd
 * rows.
 *
 * This function calculates the packed layout coordinate for an interleaved
 * layout where consecutive even and odd rows are interleaved. It takes an
 * index, the number of rows, and the number of columns as inputs and returns
 * the corresponding packed layout coordinate.
 *
 * @param idx   The index to convert to a packed layout coordinate.
 * @param rows  The number of rows in the layout.
 * @param cols  The number of columns in the layout.
 * @return      The packed layout coordinate for the given index.
 */
unsigned toPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols) {
    assert(idx < rows * cols);
    const int i = idx / cols;
    const int j = idx % cols;
    if (i % 2 == 0) {
        return i * cols + 2 * j;
    } else {
        return (i - 1) * cols + 2 * j + 1;
    }
}

/**
 * Convert a packed layout coordinate to an index for interleaved even and odd
 * rows.
 *
 * This function calculates the original index for an interleaved layout
 * where consecutive even and odd rows are interleaved. It takes a packed layout
 * coordinate, the number of rows, and the number of columns as inputs and
 * returns the corresponding index.
 *
 * @param idx   The packed layout coordinate to convert to an index.
 * @param rows  The number of rows in the layout.
 * @param cols  The number of columns in the layout.
 * @return      The index corresponding to the given packed layout coordinate.
 */
unsigned fromPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols) {
    const int i = idx / (cols * 2);
    const int j = idx % (cols * 2);
    if (j % 2 == 0) {
        return (i * 2) * cols + j / 2;
    } else {
        return (i * 2 + 1) * cols + (j - 1) / 2;
    }
}

/**
 * Check if two strings are equal while ignoring the case of characters.
 *
 * This function compares two input strings for equality, considering the case
 * of characters. It returns true if the strings are equal (ignoring case),
 * and false otherwise.
 *
 * @param str1  The first string to compare.
 * @param str2  The second string to compare.
 * @return      True if the strings are equal (ignoring case), false otherwise.
 */
bool isequalstring(const std::string &str1, const std::string &str2) {

    return str1.size() == str2.size() &&
           std::equal(str1.begin(), str1.end(), str2.begin(), str2.end(), [&](char a, char b) {
               return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
           });
}

bool g_verbose = false;
bool verbose() { return g_verbose; }
void set_verbose(bool verbose) { g_verbose = verbose; }

std::function<void(LogSeverity, const std::string &)> g_log_callback = [](LogSeverity severity,
                                                                          const std::string &msg) {
    switch (severity) {
    case LogSeverity::Warning:
        std::cerr << "tiny-dpcpp-nn warning: " << msg << std::endl;
        ;
        break;
    case LogSeverity::Error:
        std::cerr << "tiny-dpcpp-nn error: " << msg << std::endl;
        ;
        break;
    default:
        break;
    }

    if (verbose()) {
        switch (severity) {
        case LogSeverity::Debug:
            std::cerr << "tiny-dpcpp-nn debug: " << msg << std::endl;
            ;
            break;
        case LogSeverity::Info:
            std::cerr << "tiny-dpcpp-nn info: " << msg << std::endl;
            ;
            break;
        case LogSeverity::Success:
            std::cerr << "tiny-dpcpp-nn success: " << msg << std::endl;
            ;
            break;
        default:
            break;
        }
    }
};

const std::function<void(LogSeverity, const std::string &)> &log_callback() { return g_log_callback; }
void set_log_callback(const std::function<void(LogSeverity, const std::string &)> &cb) { g_log_callback = cb; }

Activation string_to_activation(const std::string &activation_name) {
    if (equals_case_insensitive(activation_name, "None")) {
        return Activation::None;
    } else if (equals_case_insensitive(activation_name, "ReLU")) {
        return Activation::ReLU;
    } else if (equals_case_insensitive(activation_name, "LeakyReLU")) {
        return Activation::LeakyReLU;
    } else if (equals_case_insensitive(activation_name, "Exponential")) {
        return Activation::Exponential;
    } else if (equals_case_insensitive(activation_name, "Sigmoid")) {
        return Activation::Sigmoid;
    } else if (equals_case_insensitive(activation_name, "Sine")) {
        return Activation::Sine;
    } else if (equals_case_insensitive(activation_name, "Squareplus")) {
        return Activation::Squareplus;
    } else if (equals_case_insensitive(activation_name, "Softplus")) {
        return Activation::Softplus;
    } else if (equals_case_insensitive(activation_name, "Tanh")) {
        return Activation::Tanh;
    }

    throw std::runtime_error{"Invalid activation name: {}"}; //, activation_name)};
}

std::string to_string(Activation activation) {
    switch (activation) {
    case Activation::None:
        return "None";
    case Activation::ReLU:
        return "ReLU";
    case Activation::LeakyReLU:
        return "LeakyReLU";
    case Activation::Exponential:
        return "Exponential";
    case Activation::Sigmoid:
        return "Sigmoid";
    case Activation::Sine:
        return "Sine";
    case Activation::Squareplus:
        return "Squareplus";
    case Activation::Softplus:
        return "Softplus";
    case Activation::Tanh:
        return "Tanh";
    default:
        throw std::runtime_error{"Invalid activation."};
    }
}

std::string to_snake_case(const std::string &str) {
    std::stringstream result;
    result << (char)std::tolower(str[0]);
    for (uint32_t i = 1; i < str.length(); ++i) {
        if (std::isupper(str[i])) {
            result << "_" << (char)std::tolower(str[i]);
        } else {
            result << str[i];
        }
    }
    return result.str();
}

std::vector<std::string> split(const std::string &text, const std::string &delim) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (true) {
        size_t end = text.find_first_of(delim, begin);
        if (end == std::string::npos) {
            result.emplace_back(text.substr(begin));
            return result;
        } else {
            result.emplace_back(text.substr(begin, end - begin));
            begin = end + 1;
        }
    }

    return result;
}

std::string to_lower(std::string str) {
    std::transform(std::begin(str), std::end(str), std::begin(str),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return str;
}

std::string to_upper(std::string str) {
    std::transform(std::begin(str), std::end(str), std::begin(str),
                   [](unsigned char c) { return (char)std::toupper(c); });
    return str;
}
