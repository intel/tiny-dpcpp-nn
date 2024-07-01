
include(CommonOptions)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/network/)

set(SOURCES_KERNEL 
    kernel_esimdbf1616none.cpp
    kernel_esimdbf1632none.cpp
    kernel_esimdbf1664none.cpp
    kernel_esimdbf16128none.cpp
    kernel_esimdfp1616none.cpp
    kernel_esimdfp1632none.cpp
    kernel_esimdfp1664none.cpp
    kernel_esimdfp16128none.cpp
    kernel_esimdbf1616relu.cpp
    kernel_esimdbf1632relu.cpp
    kernel_esimdbf1664relu.cpp
    kernel_esimdbf16128relu.cpp
    kernel_esimdfp1616relu.cpp
    kernel_esimdfp1632relu.cpp
    kernel_esimdfp1664relu.cpp
    kernel_esimdfp16128relu.cpp
    kernel_esimdbf1616sigmoid.cpp
    kernel_esimdbf1632sigmoid.cpp
    kernel_esimdbf1664sigmoid.cpp
    kernel_esimdbf16128sigmoid.cpp
    kernel_esimdfp1616sigmoid.cpp
    kernel_esimdfp1632sigmoid.cpp
    kernel_esimdfp1664sigmoid.cpp
    kernel_esimdfp16128sigmoid.cpp
)

add_library(kernels ${SOURCES_KERNEL})
target_include_directories(kernels PUBLIC ${INCLUDE_DIR})
target_link_libraries(kernels PUBLIC common)
add_sycl_to_target(TARGET kernels SOURCES ${SOURCES_KERNEL})

# set(HEADER_FILES
#     ${INCLUDE_DIR}Network.h
#     ${INCLUDE_DIR}SwiftNetMLP.h
#     ${INCLUDE_DIR}network_with_encodings.h
#     ${INCLUDE_DIR}kernel_esimd.h
# )

set(SOURCES 
    SwiftNetMLPbf1616.cpp
    SwiftNetMLPbf1632.cpp
    SwiftNetMLPbf1664.cpp
    SwiftNetMLPbf16128.cpp
    SwiftNetMLPfp1616.cpp
    SwiftNetMLPfp1632.cpp
    SwiftNetMLPfp1664.cpp
    SwiftNetMLPfp16128.cpp
)

add_library(network ${SOURCES})
target_include_directories(network PUBLIC ${INCLUDE_DIR})
target_link_libraries(network PUBLIC encodings kernels)
add_sycl_to_target(TARGET network SOURCES ${SOURCES})
# target_precompile_headers(network INTERFACE ${HEADER_FILES})
# target_precompile_headers(network REUSE_FROM encodings)