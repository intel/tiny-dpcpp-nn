# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Include headers

find_package(IntelSYCL REQUIRED)
find_package(Torch REQUIRED PATHS ${TORCH_CMAKE_PREFIX_PATH})
# Add include directories

set(LIBTORCH_DIR "${PROJECT_SOURCE_DIR}/extern/libtorch/share/cmake/")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")

find_package(IPEX REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
# Set source files for the Python bindings
set(PYBIND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tiny_dpcpp_nn/pybind_module.cpp
)

# Create the Python module
pybind11_add_module(tiny_dpcpp_nn_pybind_module ${PYBIND_SOURCES})
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_CMAKE_PREFIX_PATH}/../../lib")

# Link any required libraries
target_link_libraries(tiny_dpcpp_nn_pybind_module PRIVATE network ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${TORCH_IPEX_LIBRARIES})
target_include_directories(tiny_dpcpp_nn_pybind_module PRIVATE "${PROJECT_SOURCE_DIR}/extern/json" "${PROJECT_SOURCE_DIR}/extern/pybind11_json")
add_sycl_to_target(TARGET tiny_dpcpp_nn_pybind_module SOURCES ${PYBIND_SOURCES})

target_compile_definitions(tiny_dpcpp_nn_pybind_module
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})