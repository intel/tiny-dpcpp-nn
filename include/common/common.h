/**
 * @file common.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief File which includes all of the various functions needed everywher.
 * This file should be reworked in the future.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <array>
#include <iostream>
#include <oneapi/dpl/random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sycl/sycl.hpp>
#include <vector>

#include "vec.h"

enum class Activation {
    ReLU,
    LeakyReLU,
    Exponential,
    Sine,
    Sigmoid,
    Squareplus,
    Softplus,
    Tanh,
    None,
};

struct Context {
    Context() = default;
    virtual ~Context() {}
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;
};

/// some common math functions
namespace tinydpcppnn {
namespace math {
template <typename T> T div_round_up(T val, T divisor) { return (val + divisor - 1) / divisor; }

template <typename T> T next_multiple(T val, T divisor) { return div_round_up(val, divisor) * divisor; }

template <typename T> T previous_multiple(T val, T divisor) { return (val / divisor) * divisor; }

inline uint32_t powi(uint32_t base, uint32_t exponent) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exponent; ++i) {
        result *= base;
    }

    return result;
}

} // namespace math
} // namespace tinydpcppnn

/**
 * @brief Convert index from original matrix layout to packed layout
 *
 * @param idx Index in packed layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in packed matrix layout
 */
extern SYCL_EXTERNAL unsigned toPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols);

/**
 * @brief Convert index from packed layout to original matrix layout
 *
 * @param idx Index in original matrix layout
 * @param rows Number of rows in original matrix
 * @param cols Number of columns in original matrix
 * @return Index in original matrix layout
 */
extern SYCL_EXTERNAL unsigned fromPackedLayoutCoord(const unsigned idx, const unsigned rows, const unsigned cols);

/**
 * @brief Compare two strings case-insensitively
 *
 * @param str1 First string
 * @param str2 Second string
 * @return True if the strings are equal, false otherwise
 */
extern SYCL_EXTERNAL bool isequalstring(const std::string &str1, const std::string &str2);

namespace tinydpcppnn {
template <typename T> void format_helper(std::ostringstream &os, std::string_view &str, const T &val) {
    std::size_t bracket = str.find('{');
    if (bracket != std::string::npos) {
        std::size_t bracket_close = str.find('}', bracket + 1);
        if (bracket_close != std::string::npos) {
            os << str.substr(0, bracket) << val;
            str = str.substr(bracket_close + 1);
        } else
            throw std::invalid_argument("No closing bracket\n");
    } else
        throw std::invalid_argument("Not enough brackets for arguments\n");
};

template <typename... T> std::string format(std::string_view str, T... vals) {
    std::ostringstream os;
    (format_helper(os, str, vals), ...);
    os << str;
    return os.str();
}
} // namespace tinydpcppnn

enum class LogSeverity {
    Info,
    Debug,
    Warning,
    Error,
    Success,
};

const std::function<void(LogSeverity, const std::string &)> &log_callback();
void set_log_callback(const std::function<void(LogSeverity, const std::string &)> &callback);

template <typename... Ts> void log(LogSeverity severity, const std::string &msg, Ts &&...args) {
    log_callback()(severity, tinydpcppnn::format(msg, std::forward<Ts>(args)...)); // removed fmt, find something else
}

template <typename... Ts> void log_info(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Info, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_debug(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Debug, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_warning(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Warning, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_error(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Error, msg, std::forward<Ts>(args)...);
}
template <typename... Ts> void log_success(const std::string &msg, Ts &&...args) {
    log(LogSeverity::Success, msg, std::forward<Ts>(args)...);
}

bool verbose();
void set_verbose(bool verbose);

Activation string_to_activation(const std::string &activation_name);
std::string to_string(Activation activation);

// Hash helpers taken from https://stackoverflow.com/a/50978188
template <typename T> T xorshift(T n, int i) { return n ^ (n >> i); }

inline uint32_t distribute(uint32_t n) {
    uint32_t p = 0x55555555ul; // pattern of alternating 0 and 1
    uint32_t c = 3423571495ul; // random uneven integer constant;
    return c * xorshift(p * xorshift(n, 16), 16);
}

inline uint64_t distribute(uint64_t n) {
    uint64_t p = 0x5555555555555555ull;   // pattern of alternating 0 and 1
    uint64_t c = 17316035218449499591ull; // random uneven integer constant;
    return c * xorshift(p * xorshift(n, 32), 32);
}

template <typename T, typename S>
constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type rotl(const T n, const S i) {
    const T m = (std::numeric_limits<T>::digits - 1);
    const T c = i & m;
    return (n << c) | (n >> (((T)0 - c) & m)); // this is usually recognized by the compiler to mean rotation
}

template <typename T> size_t hash_combine(std::size_t seed, const T &v) {
    return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute(std::hash<T>{}(v));
}

std::string to_snake_case(const std::string &str);

std::vector<std::string> split(const std::string &text, const std::string &delim);

template <typename T> std::string join(const T &components, const std::string &delim) {
    std::ostringstream s;
    for (const auto &component : components) {
        if (&components[0] != &component) {
            s << delim;
        }
        s << component;
    }

    return s.str();
}

std::string to_lower(std::string str);
std::string to_upper(std::string str);

inline bool equals_case_insensitive(const std::string &str1, const std::string &str2) {
    return to_lower(str1) == to_lower(str2);
}

template <typename T> std::string type_to_string() {
    if constexpr (std::is_same<T, bool>::value)
        return "bool";
    else if constexpr (std::is_same<T, int>::value)
        return "int";
    else if constexpr (std::is_same<T, uint8_t>::value)
        return "uint8_t";
    else if constexpr (std::is_same<T, uint16_t>::value)
        return "uint16_t";
    else if constexpr (std::is_same<T, uint32_t>::value)
        return "uint32_t";
    else if constexpr (std::is_same<T, double>::value)
        return "double";
    else if constexpr (std::is_same<T, float>::value)
        return "float";
    else if constexpr (std::is_same<T, sycl::half>::value)
        return "sycl::half";
    else if constexpr (std::is_same<T, sycl::ext::oneapi::bfloat16>::value)
        return "bf16";

    return "unknown";
}