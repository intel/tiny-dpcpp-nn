# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(CommonOptions)

find_package(IntelSYCL REQUIRED)
find_package(oneDPL REQUIRED)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/common/)
set(HEADER_FILES
    ${INCLUDE_DIR}common.h
    # ${INCLUDE_DIR}DeviceMatrix.h
    # ${INCLUDE_DIR}DeviceMem.h
    # ${INCLUDE_DIR}result_check.h
    # ${INCLUDE_DIR}vec.h
)
set(SOURCES
    common.cpp
    SyclGraph.cpp
    ${HEADER_FILES}
)

add_library(common ${SOURCES})
add_library(tinydpcppnn::common ALIAS common)
set_common_options(common)
add_sycl_to_target(TARGET common SOURCES ${SOURCES})
target_link_libraries(common PUBLIC oneDPL)
target_compile_definitions(common PRIVATE -DSYCL2020_CONFORMANT_APIS)
target_include_directories(common PUBLIC ${INCLUDE_DIR})
target_compile_options(common PUBLIC -fsycl-device-code-split=per_kernel -fp-model=precise)
target_link_options(common PUBLIC -fsycl-max-parallel-link-jobs=16)
#target_precompile_headers(common PUBLIC ${HEADER_FILES})
