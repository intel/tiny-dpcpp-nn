/**
 * @file doctest_devicematrix.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief File with tests of the DeviceMatrix, DeviceMatrices, and corresponding view classes.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "DeviceMatrix.h"
#include "doctest/doctest.h"

using bf16 = sycl::ext::oneapi::bfloat16;

TEST_CASE("DeviceMatrix") {
    sycl::queue q(sycl::gpu_selector_v);
    SUBCASE("Default constructor") { CHECK_NOTHROW(DeviceMatrix<bf16> mat(1, 1, q)); }

    SUBCASE("Sizes") {
        DeviceMatrix<bf16> mat(10, 5, q);
        CHECK(mat.m() == 10);
        CHECK(mat.n() == 5);
    }

    SUBCASE("View") {
        DeviceMatrix<bf16> mat(2, 3, q);
        auto matview = mat.GetView();
        CHECK(matview.GetPointer() == mat.data());
        CHECK(matview.m() == 2);
        CHECK(matview.n() == 3);
    }
}

// TEST_CASE("DeviceMatrices") {
//     sycl::queue q(sycl::gpu_selector_v);
//     SUBCASE("Default constructor") { CHECK_NOTHROW(DeviceMatrices<bf16> mat(q, 1)); }

//     SUBCASE("AddMatrix") {
//         DeviceMatrices<bf16> mats(q, 1);
//         mats.AddMatrix(10, 5);
//         CHECK(mats.GetNumberOfMatrices() == 1);
//         CHECK_NOTHROW(mats.GetMatrix(0));
//         CHECK(mats.GetMatrix(0).size() == 50);
//     }

//     SUBCASE("AddMatrix 2 ") {
//         DeviceMatrices<bf16> mats(q, 2);
//         mats.AddMatrix(10, 5);
//         mats.AddMatrix(10, 10);
//         CHECK_THROWS(mats.AddMatrix(1, 1));
//         CHECK(mats.GetNumberOfMatrices() == 2);
//         CHECK_NOTHROW(mats.GetMatrix(0));
//         CHECK_NOTHROW(mats.GetMatrix(1));
//         CHECK(mats.GetMatrix(0).size() == 50);
//         CHECK(mats.GetMatrix(0).size() == 100);
//     }

//     SUBCASE("Matrix views") {
//         DeviceMatrices<bf16> mats(q, 1);
//         mats.AddMatrix(10, 5);
//         CHECK_NOTHROW(mats.GetViews());
//         CHECK_NOTHROW(mats.GetViews()[0]);
//         CHECK_NOTHROW(mats.GetViews()[0].GetPointer());
//         CHECK(mats.GetViews()[0].GetPointer() == mats.GetMatrix(0).data());
//     }

//     SUBCASE("Matrix views 2") {
//         DeviceMatrices<bf16> mats(q, 2);
//         mats.AddMatrix(10, 5);
//         mats.AddMatrix(10, 10);
//         CHECK_NOTHROW(mats.GetViews());
//         CHECK_NOTHROW(mats.GetViews()[0]);
//         CHECK_NOTHROW(mats.GetViews()[0].GetPointer());
//         CHECK(mats.GetViews()[0].GetPointer() == mats.GetMatrix(0).data());
//         CHECK(mats.GetViews()[1].GetPointer() == mats.GetMatrix(1).data());
//     }

//     SUBCASE("Set Matrices") {
//         DeviceMatrices<bf16> mats(q, 1);
//         mats.AddMatrix(10, 5);
//     }
// }