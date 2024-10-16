# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
find_package(MPI REQUIRED)

if(BUILD_TORCH_BENCHMARK)
    set(LIBTORCH_DIR "${PROJECT_SOURCE_DIR}/extern/libtorch/share/cmake/")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")

    find_package(IPEX REQUIRED)
    find_package(IntelSYCL REQUIRED)

    add_executable(benchmark-training-torch benchmark_training_torch.cpp)
    add_sycl_to_target(TARGET benchmark-training-torch SOURCES benchmark_training_torch.cpp)
    target_link_libraries(benchmark-training-torch PRIVATE network ${MPI_CXX_LIBRARIES} ${TORCH_IPEX_LIBRARIES})
    target_include_directories(benchmark-training-torch PRIVATE "${PROJECT_SOURCE_DIR}/include/")

    add_executable(benchmark-inference-torch benchmark_inference_torch.cpp)
    add_sycl_to_target(TARGET benchmark-inference-torch SOURCES benchmark_inference_torch.cpp)
    target_link_libraries(benchmark-inference-torch PRIVATE network ${MPI_CXX_LIBRARIES} ${TORCH_IPEX_LIBRARIES})
    target_include_directories(benchmark-inference-torch PRIVATE "${PROJECT_SOURCE_DIR}/include/")

    add_executable(benchmark-all-torch benchmark_all_torch.cpp)
    add_sycl_to_target(TARGET benchmark-all-torch SOURCES benchmark_all_torch.cpp)
    target_link_libraries(benchmark-all-torch PRIVATE network ${MPI_CXX_LIBRARIES} ${TORCH_IPEX_LIBRARIES})
    target_include_directories(benchmark-all-torch PRIVATE "${PROJECT_SOURCE_DIR}/include/")
else()
    find_package(IntelSYCL REQUIRED)
endif()


add_executable(benchmark-inference benchmark_inference.cpp)
add_sycl_to_target(TARGET benchmark-inference SOURCES benchmark_inference.cpp)
target_link_libraries(benchmark-inference PRIVATE network ${MPI_CXX_LIBRARIES})
#target_precompile_headers(benchmark-inference REUSE_FROM network)

add_executable(benchmark-training benchmark_training.cpp)
add_sycl_to_target(TARGET benchmark-training SOURCES benchmark_training.cpp)
target_link_libraries(benchmark-training PRIVATE network ${MPI_CXX_LIBRARIES})
target_include_directories(benchmark-training PRIVATE "${PROJECT_SOURCE_DIR}/include/")

add_executable(benchmark-all benchmark_all.cpp)
add_sycl_to_target(TARGET benchmark-all SOURCES benchmark_all.cpp)
target_link_libraries(benchmark-all PRIVATE network ${MPI_CXX_LIBRARIES})
target_include_directories(benchmark-all PRIVATE "${PROJECT_SOURCE_DIR}/include/")

add_executable(benchmark-encoding benchmark_encoding.cpp)
add_sycl_to_target(TARGET benchmark-encoding SOURCES benchmark_encoding.cpp)
target_link_libraries(benchmark-encoding PRIVATE network ${MPI_CXX_LIBRARIES})
target_include_directories(benchmark-encoding PRIVATE "${PROJECT_SOURCE_DIR}/include/")
