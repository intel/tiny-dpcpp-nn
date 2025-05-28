# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.23)

#the whole thing only works with icpx due to using joint_matrix
set(CMAKE_CXX_COMPILER "mpiicpx")
set(default_build_type "Release")

project(tiny-dpcpp-nn VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(CMakeDependentOption)

option(BUILD_TEST "Build tests in 'test' directory" ON)
option(BUILD_TORCH_TEST "Build tests that use torch interface (tnn_api.h) in 'test' directory" OFF)
option(BUILD_PYBIND "Build pybindings in 'dpcpp_bindings' directory" OFF)
option(BUILD_TORCH_BENCHMARK "Build benchmarks that use torch interface (tnn_api.h) in 'benchmark' directory" OFF)
option(BUILD_BENCHMARK "Build benchmarks in 'benchmark' directory" ON)
option(BUILD_DOCUMENTATION "Build the documentation which is online available" OFF)
option(BUILD_EXAMPLE "Build the examples" OFF)

set(TARGET_DEVICE "BMG" CACHE STRING "Valid values {PVC, ACM, BMG}")

set(ACM_DEVICES ACM)
set(XE_DEVICES PVC BMG)

string( TOUPPER ${TARGET_DEVICE} TARGET_DEVICE_UPPERCASE)

if (TARGET_DEVICE_UPPERCASE IN_LIST XE_DEVICES)
    message(STATUS "Building for PVC/Xe2")
    add_definitions(-DTARGET_DEVICE=0)

elseif (TARGET_DEVICE_UPPERCASE IN_LIST ACM_DEVICES)
    message(STATUS "Building for Alchemist")
    add_definitions(-DTARGET_DEVICE=1)
else()
    message(ERROR "Please set a valid device by adding -DTARGET_DEVICE=<value>, where <value> is one of  \"PVC\", \"BMG\", or \"ACM\"")
endif()


add_subdirectory(source)

enable_testing()

if(BUILD_PYBIND)
    # Include Pybind11
    add_subdirectory(extern/pybind11)
    add_subdirectory(dpcpp_bindings)
endif()

if(BUILD_DOCUMENTATION)
    add_subdirectory(docs)
endif()

if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmarks)
endif()
