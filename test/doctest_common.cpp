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

TEST_CASE("tinydpcppnn::format empty args") {

    std::string input{"This is a teststring which should work fine"};
    std::string output = tinydpcppnn::format(input);

    CHECK(input == output);
}

TEST_CASE("tinydpcppnn::format 1 string arg") {

    std::string input{"This is a teststring which should {} work fine"};
    std::string output = tinydpcppnn::format(input, "hopefully");

    CHECK(output == "This is a teststring which should hopefully work fine");
}

TEST_CASE("tinydpcppnn::format 1 int arg") {

    std::string input{"This is a teststring which should {} work fine"};
    std::string output = tinydpcppnn::format(input, 5);

    CHECK(output == "This is a teststring which should 5 work fine");
}

TEST_CASE("tinydpcppnn::format 1 double arg") {

    std::string input{"This is a teststring which should {} work fine"};
    std::string output = tinydpcppnn::format(input, 5.1);

    CHECK(output == "This is a teststring which should 5.1 work fine");
}

TEST_CASE("tinydpcppnn::format 0 arg 1 bracket") {

    std::string input{"This is a teststring which should {} work fine"};
    CHECK(tinydpcppnn::format(input) == input);
}

TEST_CASE("tinydpcppnn::format 2 arg 1 bracket") {

    std::string input{"This is a teststring which should {} work fine"};
    CHECK_THROWS_AS(tinydpcppnn::format(input, 1, 2), std::invalid_argument);
}

TEST_CASE("tinydpcppnn::format 2 arg 2 brackets") {

    std::string input{"This is a teststring which should {} work fine {}"};
    std::string output = tinydpcppnn::format(input, 1, "!");
    CHECK(output == "This is a teststring which should 1 work fine !");
}

TEST_CASE("tinydpcppnn::format 1 arg 1 incomplete bracket") {

    std::string input{"This is a teststring which should not { work fine"};
    CHECK_THROWS_AS(tinydpcppnn::format(input, 1), std::invalid_argument);
}

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
