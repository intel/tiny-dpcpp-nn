/**
 * @file doctest_devicemem.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief File with tests of the DeviceMem class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "DeviceMatrix.h"
#include "DeviceMem.h"
#include "doctest/doctest.h"

using bf16 = sycl::ext::oneapi::bfloat16;

TEST_CASE("DeviceMem") {
    sycl::queue q = queue();

    SUBCASE("Size constructor") {
        size_t size = 100;

        SUBCASE("Valid size uint8") {
            DeviceMem<uint8_t> mem(size, q);

            CHECK(mem.size() == size);
            CHECK(mem.data() != nullptr);
        }
        SUBCASE("Valid size bf16") {
            DeviceMem<bf16> mem(size, q);

            CHECK(mem.size() == size);
            CHECK(mem.data() != nullptr);
        }
        SUBCASE("Valid size float") {
            DeviceMem<float> mem(size, q);

            CHECK(mem.size() == size);
            CHECK(mem.data() != nullptr);
        }

        SUBCASE("Zero size uint8_t") { CHECK_THROWS(DeviceMem<uint8_t>(0, q)); }
        SUBCASE("Zero size bf16") { CHECK_THROWS(DeviceMem<bf16>(0, q)); }
    }
}

// Test the copy_from_host function
TEST_CASE("Testing the DeviceMem copy_from_host") {
    sycl::queue q = queue();

    SUBCASE("copy_from_host float") {
        DeviceMem<float> dm(10, q);
        dm.fill(1.0f);
        std::vector<float> data(10, 1);
        dm.copy_from_host(data);

        for (int i = 0; i < data.size(); i++) {
            CHECK(data[i] == 1);
        }
    }
}

// Test the copy_to_host function
TEST_CASE("Testing the DeviceMem copy_to_host") {
    sycl::queue q = queue();

    DeviceMem<float> dm(10, q);
    std::vector<float> data(10, 1.0f);
    dm.copy_from_host(data);
    auto data_copy = dm.copy_to_host();
    CHECK(data == data_copy);
}