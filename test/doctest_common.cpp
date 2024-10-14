/**
 * @file doctest_common.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief file to test common functions and functionalities
 * @version 0.1
 * @date 2024-01-19
 * 
 * Copyright (c) 2024 Intel Corporation
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "doctest/doctest.h"

#include "common.h"
#include "SyclGraph.h"


TEST_CASE("isequalstring 1") { CHECK(isequalstring("TEST", "TEST")); }
TEST_CASE("isequalstring 2") { CHECK(isequalstring("TesT", "TEST")); }
TEST_CASE("isequalstring 3") { CHECK(!isequalstring("TESTE", "TEST")); }
TEST_CASE("isequalstring 4") { CHECK(!isequalstring("TEST", "TESTE")); }
TEST_CASE("isequalstring 5") { CHECK(!isequalstring("", "TESTE")); }
TEST_CASE("isequalstring 6") { CHECK(isequalstring("tEsT", "TESt")); }

TEST_CASE("toPackedLayoutCoord 1") {

    int idx = 0;
    int nrows = 1;
    int ncols = 1;
    CHECK(toPackedLayoutCoord(idx, nrows, ncols) == 0);
}

TEST_CASE("toPackedLayoutCoord 2") {

    int idx = 1;
    int nrows = 2;
    int ncols = 2;
    CHECK(toPackedLayoutCoord(idx, nrows, ncols) == 2);
}

TEST_CASE("toPackedLayoutCoord 3") {

    int idx = 2;
    int nrows = 2;
    int ncols = 2;
    CHECK(toPackedLayoutCoord(idx, nrows, ncols) == 1);
}

TEST_CASE("toPackedLayoutCoord 4") {

    int idx = 4;
    int nrows = 4;
    int ncols = 2;
    CHECK(toPackedLayoutCoord(idx, nrows, ncols) == 4);
}

TEST_CASE("toPackedLayoutCoord 5") {

    int idx = 5;
    int nrows = 4;
    int ncols = 2;
    CHECK(toPackedLayoutCoord(idx, nrows, ncols) == 6);
}

TEST_CASE("toPackedLayoutCoord 6") {

    int idx = 6;
    int nrows = 4;
    int ncols = 2;
    CHECK(toPackedLayoutCoord(idx, nrows, ncols) == 5);
}

TEST_CASE("fromPackedLayoutCoord 1") {

    int idx = 0;
    int nrows = 1;
    int ncols = 1;
    CHECK(fromPackedLayoutCoord(idx, nrows, ncols) == 0);
}

TEST_CASE("fromPackedLayoutCoord 2") {

    int idx = 1;
    int nrows = 2;
    int ncols = 2;
    CHECK(fromPackedLayoutCoord(idx, nrows, ncols) == 2);
}

TEST_CASE("fromPackedLayoutCoord 3") {

    int idx = 2;
    int nrows = 2;
    int ncols = 2;
    CHECK(fromPackedLayoutCoord(idx, nrows, ncols) == 1);
}

TEST_CASE("fromPackedLayoutCoord 4") {

    int idx = 4;
    int nrows = 4;
    int ncols = 2;
    CHECK(fromPackedLayoutCoord(idx, nrows, ncols) == 4);
}

TEST_CASE("fromPackedLayoutCoord 5") {

    int idx = 5;
    int nrows = 4;
    int ncols = 2;
    CHECK(fromPackedLayoutCoord(idx, nrows, ncols) == 6);
}

TEST_CASE("fromPackedLayoutCoord 6") {

    int idx = 6;
    int nrows = 4;
    int ncols = 2;
    CHECK(fromPackedLayoutCoord(idx, nrows, ncols) == 5);
}

TEST_CASE("tinydpcppnn::SyclGraph::capture_guard"){
    constexpr int N = 1;
    auto R = sycl::range<1>( N );
    sycl::queue q{ sycl::gpu_selector_v };
    float* vec = sycl::malloc_shared<float>(N, q);

    tinydpcppnn::SyclGraph  sgraph;
    {
        auto sg = sgraph.capture_guard(&q);
        q.submit(
            [&](sycl::handler& h) {
                h.parallel_for(R, [=](sycl::id<1> i) { 
                    vec[ i ] = -2; 
                } );
            }
        );
    }
    CHECK(vec[0]==-2);
}
