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

set(IPEX_VERSION "2.1.30" CACHE STRING "The version of Intel Extension for PyTorch")
set(TARGET_DEVICE "PVC" CACHE STRING "Valid values {PVC, ARC}")

if (TARGET_DEVICE STREQUAL "PVC")
    message(STATUS "Building for PVC")
    add_definitions(-DTARGET_DEVICE=0)
elseif (TARGET_DEVICE STREQUAL "ARC")
    message(STATUS "Building for ARC")
    add_definitions(-DTARGET_DEVICE=1)
else()
    message(ERROR "Please set a valid device by adding -DTARGET_DEVICE=<value>, where <value> is either \"PVC\" or \"ARC\"")
endif()

message(WARNING "cmake build system is still WIP.")

# Check if either BUILD_PYBIND or BUILD_TORCH_TEST is ON
if(BUILD_PYBIND OR BUILD_TORCH_TEST OR BUILD_TORCH_BENCHMARK)
    # Check if libtorch directory exists
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/libtorch")
        # Notify that we are downloading the necessary libtorch files
        message(STATUS "libtorch not found in ${PROJECT_SOURCE_DIR}/extern/. Downloading...")

        # Determine the appropriate filenames based on the IPEX_VERSION
        if(IPEX_VERSION VERSION_EQUAL "2.1.30")
            set(LIBTORCH_FILENAME "libtorch-cxx11-abi-shared-with-deps-2.1.0.post2.zip")
            set(LIBINTEL_EXT_FILENAME "libintel-ext-pt-cxx11-abi-2.1.30.post0.run")
        elseif(IPEX_VERSION VERSION_EQUAL "2.1.20")
            set(LIBTORCH_FILENAME "libtorch-cxx11-abi-shared-with-deps-2.1.0.post0.zip")
            set(LIBINTEL_EXT_FILENAME "libintel-ext-pt-cxx11-abi-2.1.20+xpu.run")
        elseif(IPEX_VERSION VERSION_EQUAL "2.1.10")
            set(LIBTORCH_FILENAME "libtorch-cxx11-abi-shared-with-deps-2.1.0a0.zip")
            set(LIBINTEL_EXT_FILENAME "libintel-ext-pt-cxx11-abi-2.1.10+xpu.run")
        else()
            message(FATAL_ERROR "Unsupported IPEX_VERSION: ${IPEX_VERSION}")
        endif()

        # Set the URLs for the files to be downloaded using the determined filenames
        set(LIBTORCH_URL "https://intel-extension-for-pytorch.s3.amazonaws.com/libipex/xpu/${LIBTORCH_FILENAME}")
        set(LIBINTEL_EXT_URL "https://intel-extension-for-pytorch.s3.amazonaws.com/libipex/xpu/${LIBINTEL_EXT_FILENAME}")

        # Download libtorch
        file(DOWNLOAD "${LIBTORCH_URL}" "${CMAKE_CURRENT_BINARY_DIR}/${LIBTORCH_FILENAME}"
             STATUS download_status_libtorch)
        list(GET download_status_libtorch 0 download_result_libtorch)
        if(NOT download_result_libtorch EQUAL 0)
            message(FATAL_ERROR "Failed to download libtorch. Please download it manually with:\nwget ${LIBTORCH_URL}\nand follow install instructions for cppsdk: https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu")
        endif()

        # Download libintel extension
        file(DOWNLOAD "${LIBINTEL_EXT_URL}" "${CMAKE_CURRENT_BINARY_DIR}/${LIBINTEL_EXT_FILENAME}"
             STATUS download_status_libintel_ext)
        list(GET download_status_libintel_ext 0 download_result_libintel_ext)
        if(NOT download_result_libintel_ext EQUAL 0)
            message(FATAL_ERROR "Failed to download libintel extension. Please download it manually with:\nwget ${LIBINTEL_EXT_URL}\nand follow install instructions for cppsdk: https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu")
        endif()

        # Unzip the downloaded libtorch file
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xvf "${CMAKE_CURRENT_BINARY_DIR}/${LIBTORCH_FILENAME}"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/"
        )

        # Execute the installer for the libintel extension
        execute_process(
            COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/${LIBINTEL_EXT_FILENAME}" install "${PROJECT_SOURCE_DIR}/extern/libtorch/"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/"
        )

        # After unpacking and installation, clean up the downloaded files if needed.
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${LIBTORCH_FILENAME}")
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${LIBINTEL_EXT_FILENAME}")
    else()
        message(STATUS "libtorch already exists in ${PROJECT_SOURCE_DIR}/extern/.")
    endif()

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
