# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


include(CommonOptions)

add_subdirectory(common)

find_package(IntelSYCL REQUIRED)
find_package(oneDPL REQUIRED)
find_package(MPI REQUIRED)

add_library(encodings INTERFACE)
target_include_directories(encodings INTERFACE "${PROJECT_SOURCE_DIR}/include/encodings/"
    "${PROJECT_SOURCE_DIR}/extern/json")
target_link_libraries(encodings INTERFACE common)
target_compile_options(encodings INTERFACE -qmkl)
target_link_options(encodings INTERFACE -qmkl)
# target_precompile_headers(encodings INTERFACE encoding.h encoding_factory.h grid_interface.h grid.h identity.h spherical_harmonics.h)
# target_precompile_headers(encodings REUSE_FROM common)

add_subdirectory(network)
target_compile_options(network INTERFACE -fp-model=precise)
# target_link_options(network INTERFACE -qmkl)

add_library(loss INTERFACE)
target_include_directories(loss INTERFACE "${PROJECT_SOURCE_DIR}/include/losses/")
# target_link_libraries(loss INTERFACE encodings)
