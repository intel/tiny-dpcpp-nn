cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

#the whole thing only works with icpx due to using joint_matrix
set(CMAKE_CXX_COMPILER "mpiicpx")
set(default_build_type "Release")

set(LIBTORCH_DIR "/nfs/site/home/yuankai/libtorch/share/cmake/")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")

find_package(IPEX REQUIRED)

set(target example-app)
add_executable(${target} example-app.cpp)
target_link_libraries(${target} ${TORCH_IPEX_LIBRARIES})
list(APPEND CMAKE_CXX_FLAGS "-fsycl")

set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)