
include(CommonOptions)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/network/)
set(HEADER_FILES
    ${INCLUDE_DIR}Network.h
    ${INCLUDE_DIR}SwiftNetMLP.h
    ${INCLUDE_DIR}network_with_encodings.h
    ${INCLUDE_DIR}kernel_esimd.h
)

set(SOURCES 
    SwiftNetMLP.cpp
    ${HEADER_FILES}
)

add_library(network ${SOURCES})
target_include_directories(network PUBLIC ${INCLUDE_DIR})
target_link_libraries(network PUBLIC encodings)
add_sycl_to_target(TARGET common SOURCES ${SOURCES})
# target_precompile_headers(network INTERFACE ${HEADER_FILES})
# target_precompile_headers(network REUSE_FROM encodings)