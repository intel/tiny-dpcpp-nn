# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(IntelSYCL REQUIRED)

include(CTest)
include(${PROJECT_SOURCE_DIR}/extern/doctest/cmake/doctest.cmake)
include(CommonOptions)


add_library(test-lib STATIC main.cpp)
target_include_directories(test-lib PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
target_include_directories(test-lib PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_compile_features(test-lib PUBLIC cxx_std_20)
add_common_flags(test-lib)

add_executable(test-common doctest_common.cpp)
add_sycl_to_target(TARGET test-common SOURCES doctest_common.cpp)
target_link_libraries(test-common PRIVATE test-lib common)
doctest_discover_tests(test-common)

add_executable(test-matrix doctest_devicematrix.cpp)
add_sycl_to_target(TARGET test-matrix SOURCES doctest_devicematrix.cpp)
target_link_libraries(test-matrix PRIVATE test-lib common)
doctest_discover_tests(test-matrix)

add_executable(test-encodings doctest_encodings.cpp)
add_sycl_to_target(TARGET test-encodings SOURCES doctest_encodings.cpp)
target_link_libraries(test-encodings PRIVATE test-lib network encodings)
doctest_discover_tests(test-encodings)

add_executable(test-packing doctest_packing.cpp)
add_sycl_to_target(TARGET test-packing SOURCES doctest_packing.cpp)
target_link_libraries(test-packing PRIVATE test-lib network encodings)
doctest_discover_tests(test-packing)

add_executable(test-swiftnet doctest_swiftnet.cpp)
add_sycl_to_target(TARGET test-swiftnet SOURCES doctest_swiftnet.cpp)
target_compile_definitions(test-swiftnet PRIVATE TEST_GRAPH)
target_include_directories(test-swiftnet PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test-swiftnet PRIVATE test-lib network loss common)
doctest_discover_tests(test-swiftnet)

add_executable(test-kernel-esimd doctest_kernel_esimd.cpp)
add_sycl_to_target(TARGET test-kernel-esimd SOURCES doctest_kernel_esimd.cpp)
target_link_libraries(test-kernel-esimd PRIVATE test-lib network)
doctest_discover_tests(test-kernel-esimd)

add_executable(test-device-precision doctest_device_precision.cpp)
add_sycl_to_target(TARGET test-device-precision SOURCES doctest_device_precision.cpp)
target_link_libraries(test-device-precision PRIVATE test-lib network)
doctest_discover_tests(test-device-precision)

add_executable(test-devicemem doctest_devicemem.cpp)
add_sycl_to_target(TARGET test-devicemem SOURCES doctest_devicemem.cpp)
target_link_libraries(test-devicemem PRIVATE test-lib network)
doctest_discover_tests(test-devicemem)

add_executable(test-losses doctest_losses.cpp)
add_sycl_to_target(TARGET test-losses SOURCES doctest_losses.cpp)
target_link_libraries(test-losses PRIVATE test-lib loss common)
doctest_discover_tests(test-losses)


if(BUILD_TORCH_TEST)

    set(LIBTORCH_DIR "${PROJECT_SOURCE_DIR}/extern/libtorch/share/cmake/")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")

    find_package(IPEX REQUIRED)

    add_executable(test-torch-encodings doctest_torch_encodings.cpp)
    add_sycl_to_target(TARGET test-torch-encodings SOURCES doctest_torch_encodings.cpp)
    doctest_discover_tests(test-torch-encodings)
    target_include_directories(test-torch-encodings PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_link_libraries(test-torch-encodings PRIVATE test-lib common network encodings ${TORCH_IPEX_LIBRARIES})

    add_executable(test-torch-memory doctest_torch_memory.cpp)
    add_sycl_to_target(TARGET test-torch-memory SOURCES doctest_torch_memory.cpp)
    doctest_discover_tests(test-torch-memory)
    target_include_directories(test-torch-memory PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_link_libraries(test-torch-memory PRIVATE test-lib common network encodings ${TORCH_IPEX_LIBRARIES})

    add_executable(test-torch-network doctest_torch_network.cpp)
    add_sycl_to_target(TARGET test-torch-network SOURCES doctest_torch_network.cpp)
    doctest_discover_tests(test-torch-network)
    target_include_directories(test-torch-network PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_link_libraries(test-torch-network PRIVATE test-lib common loss network encodings ${TORCH_IPEX_LIBRARIES})

endif()