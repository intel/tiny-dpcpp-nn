/**
 * @file common_device.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief File which includes device functions.
 * Currently this file include mostly functions required in the grid encoding.
 * Those functions should be moved into the grid encoding in the near future.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "common.h"

#include <oneapi/dpl/random>
#include <sycl/sycl.hpp>

inline float logistic(const float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// Swap this to get rid of onedpl
inline float random_val(uint32_t seed, uint32_t idx) {

    // Create minstd_rand engine
    oneapi::dpl::minstd_rand engine(seed, idx);
    oneapi::dpl::uniform_real_distribution<float> distr;
    // Generate float random number
    return distr(engine);
}
template <typename T, typename ARRAY_T>
void sh_enc(uint32_t degree, float x, float y, float z, ARRAY_T &data_out, int offset) {

    // Let compiler figure out how to sequence/reorder these calculations w.r.t.
    // branches
    float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;

    // SH polynomials generated using scripts/gen_sh.py based on the recurrence
    // relations in appendix A1 of
    // https://www.ppsloan.org/publications/StupidSH36.pdf
    data_out[0 + offset] = (T)(0.28209479177387814f); // 1/(2*sqrt(pi))
    if (degree <= 1) {
        return;
    }
    data_out[1 + offset] = (T)(-0.48860251190291987f * y); // -sqrt(3)*y/(2*sqrt(pi))
    data_out[2 + offset] = (T)(0.48860251190291987f * z);  // sqrt(3)*z/(2*sqrt(pi))
    data_out[3 + offset] = (T)(-0.48860251190291987f * x); // -sqrt(3)*x/(2*sqrt(pi))
    if (degree <= 2) {
        return;
    }
    data_out[4 + offset] = (T)(1.0925484305920792f * xy);                         // sqrt(15)*xy/(2*sqrt(pi))
    data_out[5 + offset] = (T)(-1.0925484305920792f * yz);                        // -sqrt(15)*yz/(2*sqrt(pi))
    data_out[6 + offset] = (T)(0.94617469575755997f * z2 - 0.31539156525251999f); // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    data_out[7 + offset] = (T)(-1.0925484305920792f * xz);                        // -sqrt(15)*xz/(2*sqrt(pi))
    data_out[8 + offset] =
        (T)(0.54627421529603959f * x2 - 0.54627421529603959f * y2); // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if (degree <= 3) {
        return;
    }
    data_out[9 + offset] = (T)(0.59004358992664352f * y * (-3.0f * x2 + y2));   // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    data_out[10 + offset] = (T)(2.8906114426405538f * xy * z);                  // sqrt(105)*xy*z/(2*sqrt(pi))
    data_out[11 + offset] = (T)(0.45704579946446572f * y * (1.0f - 5.0f * z2)); // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
    data_out[12 + offset] = (T)(0.3731763325901154f * z * (5.0f * z2 - 3.0f));  // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
    data_out[13 + offset] = (T)(0.45704579946446572f * x * (1.0f - 5.0f * z2)); // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
    data_out[14 + offset] = (T)(1.4453057213202769f * z * (x2 - y2));           // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
    data_out[15 + offset] = (T)(0.59004358992664352f * x * (-x2 + 3.0f * y2));  // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    if (degree <= 4) {
        return;
    }
    data_out[16 + offset] = (T)(2.5033429417967046f * xy * (x2 - y2)); // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
    data_out[17 + offset] =
        (T)(1.7701307697799304f * yz * (-3.0f * x2 + y2)); // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
    data_out[18 + offset] = (T)(0.94617469575756008f * xy * (7.0f * z2 - 1.0f)); // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
    data_out[19 + offset] =
        (T)(0.66904654355728921f * yz * (3.0f - 7.0f * z2)); // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
    data_out[20 + offset] = (T)(-3.1735664074561294f * z2 + 3.7024941420321507f * z4 +
                                0.31735664074561293f); // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
    data_out[21 + offset] =
        (T)(0.66904654355728921f * xz * (3.0f - 7.0f * z2)); // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
    data_out[22 + offset] =
        (T)(0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f)); // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
    data_out[23 + offset] =
        (T)(1.7701307697799304f * xz * (-x2 + 3.0f * y2)); // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
    data_out[24 + offset] = (T)(-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 +
                                0.62583573544917614f * y4); // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    if (degree <= 5) {
        return;
    }
    data_out[25 + offset] =
        (T)(0.65638205684017015f * y *
            (10.0f * x2 * y2 - 5.0f * x4 - y4)); // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    data_out[26 + offset] = (T)(8.3026492595241645f * xy * z * (x2 - y2)); // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
    data_out[27 + offset] = (T)(-0.48923829943525038f * y * (3.0f * x2 - y2) *
                                (9.0f * z2 - 1.0f)); // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    data_out[28 + offset] =
        (T)(4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f)); // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
    data_out[29 + offset] = (T)(0.45294665119569694f * y *
                                (14.0f * z2 - 21.0f * z4 - 1.0f)); // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    data_out[30 + offset] = (T)(0.1169503224534236f * z *
                                (-70.0f * z2 + 63.0f * z4 + 15.0f)); // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
    data_out[31 + offset] = (T)(0.45294665119569694f * x *
                                (14.0f * z2 - 21.0f * z4 - 1.0f)); // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    data_out[32 + offset] =
        (T)(2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f)); // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
    data_out[33 + offset] = (T)(-0.48923829943525038f * x * (x2 - 3.0f * y2) *
                                (9.0f * z2 - 1.0f)); // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
    data_out[34 + offset] =
        (T)(2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4)); // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    data_out[35 + offset] =
        (T)(0.65638205684017015f * x *
            (10.0f * x2 * y2 - x4 - 5.0f * y4)); // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    if (degree <= 6) {
        return;
    }
    data_out[36 + offset] =
        (T)(1.3663682103838286f * xy *
            (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4)); // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    data_out[37 + offset] =
        (T)(2.3666191622317521f * yz *
            (10.0f * x2 * y2 - 5.0f * x4 - y4)); // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    data_out[38 + offset] = (T)(2.0182596029148963f * xy * (x2 - y2) *
                                (11.0f * z2 - 1.0f)); // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    data_out[39 + offset] = (T)(-0.92120525951492349f * yz * (3.0f * x2 - y2) *
                                (11.0f * z2 - 3.0f)); // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    data_out[40 + offset] = (T)(0.92120525951492349f * xy *
                                (-18.0f * z2 + 33.0f * z4 + 1.0f)); // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    data_out[41 + offset] = (T)(0.58262136251873131f * yz *
                                (30.0f * z2 - 33.0f * z4 - 5.0f)); // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    data_out[42 + offset] = (T)(6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 -
                                0.31784601133814211f); // sqrt(13)*(105*z2 - 315*z4 +
                                                       // 231*z6 - 5)/(32*sqrt(pi))
    data_out[43 + offset] = (T)(0.58262136251873131f * xz *
                                (30.0f * z2 - 33.0f * z4 - 5.0f)); // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    data_out[44 + offset] =
        (T)(0.46060262975746175f * (x2 - y2) *
            (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f)); // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2
                                                                   // + 1)/(64*sqrt(pi))
    data_out[45 + offset] = (T)(-0.92120525951492349f * xz * (x2 - 3.0f * y2) *
                                (11.0f * z2 - 3.0f)); // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
    data_out[46 + offset] =
        (T)(0.50456490072872406f * (11.0f * z2 - 1.0f) *
            (-6.0f * x2 * y2 + x4 + y4)); // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    data_out[47 + offset] =
        (T)(2.3666191622317521f * xz *
            (10.0f * x2 * y2 - x4 - 5.0f * y4)); // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    data_out[48 + offset] =
        (T)(10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 -
            0.6831841051919143f * y6); // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    if (degree <= 7) {
        return;
    }
    data_out[49 + offset] =
        (T)(0.70716273252459627f * y *
            (-21.0f * x2 * y4 + 35.0f * x4 * y2 - 7.0f * x6 + y6)); // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 -
                                                                    // 7*x6 + y6)/(64*sqrt(pi))
    data_out[50 + offset] =
        (T)(5.2919213236038001f * xy * z * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4)); // 3*sqrt(10010)*xy*z*(-10*x2*y2
                                                                                        // + 3*x4 + 3*y4)/(32*sqrt(pi))
    data_out[51 + offset] = (T)(-0.51891557872026028f * y * (13.0f * z2 - 1.0f) *
                                (-10.0f * x2 * y2 + 5.0f * x4 + y4)); // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 +
                                                                      // 5*x4 + y4)/(64*sqrt(pi))
    data_out[52 + offset] = (T)(4.1513246297620823f * xy * z * (x2 - y2) *
                                (13.0f * z2 - 3.0f)); // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    data_out[53 + offset] =
        (T)(-0.15645893386229404f * y * (3.0f * x2 - y2) *
            (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f)); // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) -
                                                                     // 27*z2 + 3)/(64*sqrt(pi))
    data_out[54 + offset] =
        (T)(0.44253269244498261f * xy * z *
            (-110.0f * z2 + 143.0f * z4 + 15.0f)); // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    data_out[55 + offset] = (T)(0.090331607582517306f * y *
                                (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 +
                                 5.0f)); // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    data_out[56 + offset] = (T)(0.068284276912004949f * z *
                                (315.0f * z2 - 693.0f * z4 + 429.0f * z6 -
                                 35.0f)); // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
    data_out[57 + offset] = (T)(0.090331607582517306f * x *
                                (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 +
                                 5.0f)); // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    data_out[58 + offset] =
        (T)(0.07375544874083044f * z * (x2 - y2) *
            (143.0f * z2 * (3.0f * z2 - 1.0f) - 187.0f * z2 + 45.0f)); // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 -
                                                                       // 1) - 187*z2 + 45)/(64*sqrt(pi))
    data_out[59 + offset] =
        (T)(-0.15645893386229404f * x * (x2 - 3.0f * y2) *
            (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f)); // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) -
                                                                     // 27*z2 + 3)/(64*sqrt(pi))
    data_out[60 + offset] = (T)(1.0378311574405206f * z * (13.0f * z2 - 3.0f) *
                                (-6.0f * x2 * y2 + x4 + y4)); // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2
                                                              // + x4 + y4)/(32*sqrt(pi))
    data_out[61 + offset] = (T)(-0.51891557872026028f * x * (13.0f * z2 - 1.0f) *
                                (-10.0f * x2 * y2 + x4 + 5.0f * y4)); // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 +
                                                                      // x4 + 5*y4)/(64*sqrt(pi))
    data_out[62 + offset] =
        (T)(2.6459606618019f * z * (15.0f * x2 * y4 - 15.0f * x4 * y2 + x6 - y6)); // 3*sqrt(10010)*z*(15*x2*y4 -
                                                                                   // 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    data_out[63 + offset] =
        (T)(0.70716273252459627f * x *
            (-35.0f * x2 * y4 + 21.0f * x4 * y2 - x6 + 7.0f * y6)); // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6
                                                                    // + 7*y6)/(64*sqrt(pi))4*y2 - x6
                                                                    // + 7*y6)/(64*sqrt(pi))
}

template <typename T, typename ARRAY_T> void sh_enc(uint32_t degree, float x, float y, float z, ARRAY_T &data_out) {
    // Let compiler figure out how to sequence/reorder these calculations w.r.t.
    // branches
    float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;

    // SH polynomials generated using scripts/gen_sh.py based on the recurrence
    // relations in appendix A1 of
    // https://www.ppsloan.org/publications/StupidSH36.pdf
    data_out(0) = (T)(0.28209479177387814f); // 1/(2*sqrt(pi))
    if (degree <= 1) {
        return;
    }
    data_out(1) = (T)(-0.48860251190291987f * y); // -sqrt(3)*y/(2*sqrt(pi))
    data_out(2) = (T)(0.48860251190291987f * z);  // sqrt(3)*z/(2*sqrt(pi))
    data_out(3) = (T)(-0.48860251190291987f * x); // -sqrt(3)*x/(2*sqrt(pi))
    if (degree <= 2) {
        return;
    }
    data_out(4) = (T)(1.0925484305920792f * xy);                              // sqrt(15)*xy/(2*sqrt(pi))
    data_out(5) = (T)(-1.0925484305920792f * yz);                             // -sqrt(15)*yz/(2*sqrt(pi))
    data_out(6) = (T)(0.94617469575755997f * z2 - 0.31539156525251999f);      // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    data_out(7) = (T)(-1.0925484305920792f * xz);                             // -sqrt(15)*xz/(2*sqrt(pi))
    data_out(8) = (T)(0.54627421529603959f * x2 - 0.54627421529603959f * y2); // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if (degree <= 3) {
        return;
    }
    data_out(9) = (T)(0.59004358992664352f * y * (-3.0f * x2 + y2));   // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    data_out(10) = (T)(2.8906114426405538f * xy * z);                  // sqrt(105)*xy*z/(2*sqrt(pi))
    data_out(11) = (T)(0.45704579946446572f * y * (1.0f - 5.0f * z2)); // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
    data_out(12) = (T)(0.3731763325901154f * z * (5.0f * z2 - 3.0f));  // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
    data_out(13) = (T)(0.45704579946446572f * x * (1.0f - 5.0f * z2)); // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
    data_out(14) = (T)(1.4453057213202769f * z * (x2 - y2));           // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
    data_out(15) = (T)(0.59004358992664352f * x * (-x2 + 3.0f * y2));  // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    if (degree <= 4) {
        return;
    }
    data_out(16) = (T)(2.5033429417967046f * xy * (x2 - y2));           // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
    data_out(17) = (T)(1.7701307697799304f * yz * (-3.0f * x2 + y2));   // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
    data_out(18) = (T)(0.94617469575756008f * xy * (7.0f * z2 - 1.0f)); // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
    data_out(19) = (T)(0.66904654355728921f * yz * (3.0f - 7.0f * z2)); // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
    data_out(20) = (T)(-3.1735664074561294f * z2 + 3.7024941420321507f * z4 +
                       0.31735664074561293f);                           // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
    data_out(21) = (T)(0.66904654355728921f * xz * (3.0f - 7.0f * z2)); // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
    data_out(22) =
        (T)(0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f));   // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
    data_out(23) = (T)(1.7701307697799304f * xz * (-x2 + 3.0f * y2)); // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
    data_out(24) = (T)(-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 +
                       0.62583573544917614f * y4); // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    if (degree <= 5) {
        return;
    }
    data_out(25) = (T)(0.65638205684017015f * y *
                       (10.0f * x2 * y2 - 5.0f * x4 - y4));       // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    data_out(26) = (T)(8.3026492595241645f * xy * z * (x2 - y2)); // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
    data_out(27) = (T)(-0.48923829943525038f * y * (3.0f * x2 - y2) *
                       (9.0f * z2 - 1.0f)); // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    data_out(28) = (T)(4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f)); // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
    data_out(29) = (T)(0.45294665119569694f * y *
                       (14.0f * z2 - 21.0f * z4 - 1.0f)); // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    data_out(30) = (T)(0.1169503224534236f * z *
                       (-70.0f * z2 + 63.0f * z4 + 15.0f)); // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
    data_out(31) = (T)(0.45294665119569694f * x *
                       (14.0f * z2 - 21.0f * z4 - 1.0f)); // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    data_out(32) =
        (T)(2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f)); // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
    data_out(33) = (T)(-0.48923829943525038f * x * (x2 - 3.0f * y2) *
                       (9.0f * z2 - 1.0f)); // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
    data_out(34) =
        (T)(2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4)); // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    data_out(35) = (T)(0.65638205684017015f * x *
                       (10.0f * x2 * y2 - x4 - 5.0f * y4)); // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    if (degree <= 6) {
        return;
    }
    data_out(36) =
        (T)(1.3663682103838286f * xy *
            (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4)); // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    data_out(37) = (T)(2.3666191622317521f * yz *
                       (10.0f * x2 * y2 - 5.0f * x4 - y4)); // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    data_out(38) = (T)(2.0182596029148963f * xy * (x2 - y2) *
                       (11.0f * z2 - 1.0f)); // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    data_out(39) = (T)(-0.92120525951492349f * yz * (3.0f * x2 - y2) *
                       (11.0f * z2 - 3.0f)); // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    data_out(40) = (T)(0.92120525951492349f * xy *
                       (-18.0f * z2 + 33.0f * z4 + 1.0f)); // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    data_out(41) = (T)(0.58262136251873131f * yz *
                       (30.0f * z2 - 33.0f * z4 - 5.0f)); // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    data_out(42) = (T)(6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 -
                       0.31784601133814211f); // sqrt(13)*(105*z2 - 315*z4 +
                                              // 231*z6 - 5)/(32*sqrt(pi))
    data_out(43) = (T)(0.58262136251873131f * xz *
                       (30.0f * z2 - 33.0f * z4 - 5.0f)); // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    data_out(44) =
        (T)(0.46060262975746175f * (x2 - y2) *
            (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f)); // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2
                                                                   // + 1)/(64*sqrt(pi))
    data_out(45) = (T)(-0.92120525951492349f * xz * (x2 - 3.0f * y2) *
                       (11.0f * z2 - 3.0f)); // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
    data_out(46) = (T)(0.50456490072872406f * (11.0f * z2 - 1.0f) *
                       (-6.0f * x2 * y2 + x4 + y4)); // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    data_out(47) = (T)(2.3666191622317521f * xz *
                       (10.0f * x2 * y2 - x4 - 5.0f * y4)); // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    data_out(48) = (T)(10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 -
                       0.6831841051919143f * y6); // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    if (degree <= 7) {
        return;
    }
    data_out(49) = (T)(0.70716273252459627f * y *
                       (-21.0f * x2 * y4 + 35.0f * x4 * y2 - 7.0f * x6 + y6)); // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 -
                                                                               // 7*x6 + y6)/(64*sqrt(pi))
    data_out(50) =
        (T)(5.2919213236038001f * xy * z * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4)); // 3*sqrt(10010)*xy*z*(-10*x2*y2
                                                                                        // + 3*x4 + 3*y4)/(32*sqrt(pi))
    data_out(51) = (T)(-0.51891557872026028f * y * (13.0f * z2 - 1.0f) *
                       (-10.0f * x2 * y2 + 5.0f * x4 + y4)); // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 +
                                                             // y4)/(64*sqrt(pi))
    data_out(52) = (T)(4.1513246297620823f * xy * z * (x2 - y2) *
                       (13.0f * z2 - 3.0f)); // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    data_out(53) =
        (T)(-0.15645893386229404f * y * (3.0f * x2 - y2) *
            (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f)); // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) -
                                                                     // 27*z2 + 3)/(64*sqrt(pi))
    data_out(54) = (T)(0.44253269244498261f * xy * z *
                       (-110.0f * z2 + 143.0f * z4 + 15.0f)); // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    data_out(55) = (T)(0.090331607582517306f * y *
                       (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 +
                        5.0f)); // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    data_out(56) = (T)(0.068284276912004949f * z *
                       (315.0f * z2 - 693.0f * z4 + 429.0f * z6 -
                        35.0f)); // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
    data_out(57) = (T)(0.090331607582517306f * x *
                       (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 +
                        5.0f)); // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    data_out(58) = (T)(0.07375544874083044f * z * (x2 - y2) *
                       (143.0f * z2 * (3.0f * z2 - 1.0f) - 187.0f * z2 + 45.0f)); // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2
                                                                                  // - 1) - 187*z2 + 45)/(64*sqrt(pi))
    data_out(59) =
        (T)(-0.15645893386229404f * x * (x2 - 3.0f * y2) *
            (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f)); // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) -
                                                                     // 27*z2 + 3)/(64*sqrt(pi))
    data_out(60) = (T)(1.0378311574405206f * z * (13.0f * z2 - 3.0f) *
                       (-6.0f * x2 * y2 + x4 + y4)); // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2
                                                     // + x4 + y4)/(32*sqrt(pi))
    data_out(61) = (T)(-0.51891557872026028f * x * (13.0f * z2 - 1.0f) *
                       (-10.0f * x2 * y2 + x4 + 5.0f * y4)); // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 +
                                                             // x4 + 5*y4)/(64*sqrt(pi))
    data_out(62) =
        (T)(2.6459606618019f * z * (15.0f * x2 * y4 - 15.0f * x4 * y2 + x6 - y6)); // 3*sqrt(10010)*z*(15*x2*y4 -
                                                                                   // 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    data_out(63) =
        (T)(0.70716273252459627f * x *
            (-35.0f * x2 * y4 + 21.0f * x4 * y2 - x6 + 7.0f * y6)); // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6
                                                                    // + 7*y6)/(64*sqrt(pi))
}

template <typename T, typename ARRAY_T>
tnn::vec3 sh_enc_grad(uint32_t degree, float x, float y, float z, const ARRAY_T &dL_dy) {
    // Let compiler figure out how to sequence/reorder these calculations w.r.t.
    // branches
    float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;

    tnn::vec3 d(0.0f);

    // d.x += (float)dL_dy(0) * (0);                                // 0
    // d.y += (float)dL_dy(0) * (0);                                // 0
    // d.z += (float)dL_dy(0) * (0);                                // 0
    if (degree <= 1) {
        return d;
    }
    // d.x += (float)dL_dy(1) * (0);                                // 0
    d.y += (float)dL_dy(1) * (-0.48860251190291992); // -sqrt(3)/(2*sqrt(pi))
    // d.z += (float)dL_dy(1) * (0);                                // 0
    // d.x += (float)dL_dy(2) * (0);                                // 0
    // d.y += (float)dL_dy(2) * (0);                                // 0
    d.z += (float)dL_dy(2) * (0.48860251190291992);  // sqrt(3)/(2*sqrt(pi))
    d.x += (float)dL_dy(3) * (-0.48860251190291992); // -sqrt(3)/(2*sqrt(pi))
    // d.y += (float)dL_dy(3) * (0);                                // 0
    // d.z += (float)dL_dy(3) * (0);                                // 0
    if (degree <= 2) {
        return d;
    }
    d.x += (float)dL_dy(4) * (1.0925484305920792 * y); // sqrt(15)*y/(2*sqrt(pi))
    d.y += (float)dL_dy(4) * (1.0925484305920792 * x); // sqrt(15)*x/(2*sqrt(pi))
    // d.z += (float)dL_dy(4) * (0);                                // 0
    // d.x += (float)dL_dy(5) * (0);                                // 0
    d.y += (float)dL_dy(5) * (-1.0925484305920792 * z); // -sqrt(15)*z/(2*sqrt(pi))
    d.z += (float)dL_dy(5) * (-1.0925484305920792 * y); // -sqrt(15)*y/(2*sqrt(pi))
    // d.x += (float)dL_dy(6) * (0);                                // 0
    // d.y += (float)dL_dy(6) * (0);                                // 0
    d.z += (float)dL_dy(6) * (1.8923493915151202 * z);  // 3*sqrt(5)*z/(2*sqrt(pi))
    d.x += (float)dL_dy(7) * (-1.0925484305920792 * z); // -sqrt(15)*z/(2*sqrt(pi))
    // d.y += (float)dL_dy(7) * (0);                                // 0
    d.z += (float)dL_dy(7) * (-1.0925484305920792 * x); // -sqrt(15)*x/(2*sqrt(pi))
    d.x += (float)dL_dy(8) * (1.0925484305920792 * x);  // sqrt(15)*x/(2*sqrt(pi))
    d.y += (float)dL_dy(8) * (-1.0925484305920792 * y); // -sqrt(15)*y/(2*sqrt(pi))
    // d.z += (float)dL_dy(8) * (0);                                // 0
    if (degree <= 3) {
        return d;
    }
    d.x += (float)dL_dy(9) * (-3.5402615395598609 * xy);                           // -3*sqrt(70)*xy/(4*sqrt(pi))
    d.y += (float)dL_dy(9) * (-1.7701307697799304 * x2 + 1.7701307697799304 * y2); // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
    // d.z += (float)dL_dy(9) * (0);                                // 0
    d.x += (float)dL_dy(10) * (2.8906114426405538 * yz); // sqrt(105)*yz/(2*sqrt(pi))
    d.y += (float)dL_dy(10) * (2.8906114426405538 * xz); // sqrt(105)*xz/(2*sqrt(pi))
    d.z += (float)dL_dy(10) * (2.8906114426405538 * xy); // sqrt(105)*xy/(2*sqrt(pi))
    // d.x += (float)dL_dy(11) * (0);                               // 0
    d.y += (float)dL_dy(11) * (0.45704579946446572 - 2.2852289973223288 * z2); // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
    d.z += (float)dL_dy(11) * (-4.5704579946446566 * yz);                      // -5*sqrt(42)*yz/(4*sqrt(pi))
    // d.x += (float)dL_dy(12) * (0);                               // 0
    // d.y += (float)dL_dy(12) * (0);                               // 0
    d.z += (float)dL_dy(12) * (5.597644988851731 * z2 - 1.1195289977703462);   // 3*sqrt(7)*(5*z2 - 1)/(4*sqrt(pi))
    d.x += (float)dL_dy(13) * (0.45704579946446572 - 2.2852289973223288 * z2); // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
    // d.y += (float)dL_dy(13) * (0);                               // 0
    d.z += (float)dL_dy(13) * (-4.5704579946446566 * xz);                          // -5*sqrt(42)*xz/(4*sqrt(pi))
    d.x += (float)dL_dy(14) * (2.8906114426405538 * xz);                           // sqrt(105)*xz/(2*sqrt(pi))
    d.y += (float)dL_dy(14) * (-2.8906114426405538 * yz);                          // -sqrt(105)*yz/(2*sqrt(pi))
    d.z += (float)dL_dy(14) * (1.4453057213202769 * x2 - 1.4453057213202769 * y2); // sqrt(105)*(x2 - y2)/(4*sqrt(pi))
    d.x +=
        (float)dL_dy(15) * (-1.7701307697799304 * x2 + 1.7701307697799304 * y2); // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
    d.y += (float)dL_dy(15) * (3.5402615395598609 * xy);                         // 3*sqrt(70)*xy/(4*sqrt(pi))
    // d.z += (float)dL_dy(15) * (0);                               // 0
    if (degree <= 4) {
        return d;
    }
    d.x += (float)dL_dy(16) * (2.5033429417967046 * y * (3.0 * x2 - y2)); // 3*sqrt(35)*y*(3*x2 - y2)/(4*sqrt(pi))
    d.y += (float)dL_dy(16) * (2.5033429417967046 * x * (x2 - 3.0 * y2)); // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
    // d.z += (float)dL_dy(16) * (0);                               // 0
    d.x += (float)dL_dy(17) * (-10.620784618679583 * xy * z);               // -9*sqrt(70)*xy*z/(4*sqrt(pi))
    d.y += (float)dL_dy(17) * (5.3103923093397913 * z * (-x2 + y2));        // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
    d.z += (float)dL_dy(17) * (1.7701307697799304 * y * (-3.0 * x2 + y2));  // 3*sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    d.x += (float)dL_dy(18) * (0.94617469575756008 * y * (7.0 * z2 - 1.0)); // 3*sqrt(5)*y*(7*z2 - 1)/(4*sqrt(pi))
    d.y += (float)dL_dy(18) * (0.94617469575756008 * x * (7.0 * z2 - 1.0)); // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
    d.z += (float)dL_dy(18) * (13.246445740605839 * xy * z);                // 21*sqrt(5)*xy*z/(2*sqrt(pi))
    // d.x += (float)dL_dy(19) * (0);                               // 0
    d.y += (float)dL_dy(19) * (0.66904654355728921 * z * (3.0 - 7.0 * z2)); // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
    d.z += (float)dL_dy(19) * (2.0071396306718676 * y * (1.0 - 7.0 * z2));  // 9*sqrt(10)*y*(1 - 7*z2)/(8*sqrt(pi))
    // d.x += (float)dL_dy(20) * (0);                               // 0
    // d.y += (float)dL_dy(20) * (0);                               // 0
    d.z += (float)dL_dy(20) * (14.809976568128603 * z * z2 - 6.3471328149122579 * z); // (105*z**3 - 45*z)/(4*sqrt(pi))
    d.x += (float)dL_dy(21) * (0.66904654355728921 * z * (3.0 - 7.0 * z2)); // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
    // d.y += (float)dL_dy(21) * (0);                               // 0
    d.z += (float)dL_dy(21) * (2.0071396306718676 * x * (1.0 - 7.0 * z2));  // 9*sqrt(10)*x*(1 - 7*z2)/(8*sqrt(pi))
    d.x += (float)dL_dy(22) * (0.94617469575756008 * x * (7.0 * z2 - 1.0)); // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
    d.y += (float)dL_dy(22) * (0.94617469575756008 * y * (1.0 - 7.0 * z2)); // 3*sqrt(5)*y*(1 - 7*z2)/(4*sqrt(pi))
    d.z += (float)dL_dy(22) * (6.6232228703029197 * z * (x2 - y2));         // 21*sqrt(5)*z*(x2 - y2)/(4*sqrt(pi))
    d.x += (float)dL_dy(23) * (5.3103923093397913 * z * (-x2 + y2));        // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
    d.y += (float)dL_dy(23) * (10.620784618679583 * xy * z);                // 9*sqrt(70)*xy*z/(4*sqrt(pi))
    d.z += (float)dL_dy(23) * (1.7701307697799304 * x * (-x2 + 3.0 * y2));  // 3*sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    d.x += (float)dL_dy(24) * (2.5033429417967046 * x * (x2 - 3.0 * y2));   // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
    d.y += (float)dL_dy(24) * (2.5033429417967046 * y * (-3.0 * x2 + y2));  // 3*sqrt(35)*y*(-3*x2 + y2)/(4*sqrt(pi))
    // d.z += (float)dL_dy(24) * (0);                               // 0
    if (degree <= 5) {
        return d;
    }
    d.x += (float)dL_dy(25) * (13.127641136803401 * xy * (-x2 + y2)); // 15*sqrt(154)*xy*(-x2 + y2)/(8*sqrt(pi))
    d.y += (float)dL_dy(25) * (19.6914617052051 * x2 * y2 - 3.2819102842008503 * x4 -
                               3.2819102842008503 * y4); // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    // d.z += (float)dL_dy(25) * (0);                               // 0
    d.x += (float)dL_dy(26) * (8.3026492595241645 * yz * (3.0 * x2 - y2));  // 3*sqrt(385)*yz*(3*x2 - y2)/(4*sqrt(pi))
    d.y += (float)dL_dy(26) * (8.3026492595241645 * xz * (x2 - 3.0 * y2));  // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
    d.z += (float)dL_dy(26) * (8.3026492595241645 * xy * (x2 - y2));        // 3*sqrt(385)*xy*(x2 - y2)/(4*sqrt(pi))
    d.x += (float)dL_dy(27) * (2.9354297966115022 * xy * (1.0 - 9.0 * z2)); // 3*sqrt(770)*xy*(1 - 9*z2)/(16*sqrt(pi))
    d.y += (float)dL_dy(27) *
           (-1.4677148983057511 * (x2 - y2) * (9.0 * z2 - 1.0)); // -3*sqrt(770)*(x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    d.z += (float)dL_dy(27) * (8.8062893898345074 * yz * (-3.0 * x2 + y2)); // 9*sqrt(770)*yz*(-3*x2 + y2)/(16*sqrt(pi))
    d.x += (float)dL_dy(28) * (4.7935367849733241 * yz * (3.0 * z2 - 1.0)); // sqrt(1155)*yz*(3*z2 - 1)/(4*sqrt(pi))
    d.y += (float)dL_dy(28) * (4.7935367849733241 * xz * (3.0 * z2 - 1.0)); // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
    d.z += (float)dL_dy(28) * (4.7935367849733241 * xy * (9.0 * z2 - 1.0)); // sqrt(1155)*xy*(9*z2 - 1)/(4*sqrt(pi))
    // d.x += (float)dL_dy(29) * (0);                               // 0
    d.y += (float)dL_dy(29) * (6.3412531167397574 * z2 - 9.5118796751096362 * z4 -
                               0.45294665119569694); // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    d.z += (float)dL_dy(29) * (12.682506233479513 * yz * (1.0 - 3.0 * z2)); // 7*sqrt(165)*yz*(1 - 3*z2)/(4*sqrt(pi))
    // d.x += (float)dL_dy(30) * (0);                               // 0
    // d.y += (float)dL_dy(30) * (0);                               // 0
    d.z += (float)dL_dy(30) * (-24.559567715218954 * z2 + 36.839351572828434 * z4 +
                               1.754254836801354); // 15*sqrt(11)*(-14*z2 + 21*z4 + 1)/(16*sqrt(pi))
    d.x += (float)dL_dy(31) * (6.3412531167397574 * z2 - 9.5118796751096362 * z4 -
                               0.45294665119569694); // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    // d.y += (float)dL_dy(31) * (0);                               // 0
    d.z += (float)dL_dy(31) * (12.682506233479513 * xz * (1.0 - 3.0 * z2)); // 7*sqrt(165)*xz*(1 - 3*z2)/(4*sqrt(pi))
    d.x += (float)dL_dy(32) * (4.7935367849733241 * xz * (3.0 * z2 - 1.0)); // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
    d.y += (float)dL_dy(32) * (4.7935367849733241 * yz * (1.0 - 3.0 * z2)); // sqrt(1155)*yz*(1 - 3*z2)/(4*sqrt(pi))
    d.z += (float)dL_dy(32) *
           (2.3967683924866621 * (x2 - y2) * (9.0 * z2 - 1.0)); // sqrt(1155)*(x2 - y2)*(9*z2 - 1)/(8*sqrt(pi))
    d.x += (float)dL_dy(33) * (-13.209434084751759 * x2 * z2 + 1.4677148983057511 * x2 + 13.209434084751759 * y2 * z2 -
                               1.4677148983057511 * y2); // 3*sqrt(770)*(-9*x2*z2 + x2 + 9*y2*z2 - y2)/(32*sqrt(pi))
    d.y += (float)dL_dy(33) * (2.9354297966115022 * xy * (9.0 * z2 - 1.0)); // 3*sqrt(770)*xy*(9*z2 - 1)/(16*sqrt(pi))
    d.z += (float)dL_dy(33) * (8.8062893898345074 * xz * (-x2 + 3.0 * y2)); // 9*sqrt(770)*xz*(-x2 + 3*y2)/(16*sqrt(pi))
    d.x += (float)dL_dy(34) * (8.3026492595241645 * xz * (x2 - 3.0 * y2));  // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
    d.y += (float)dL_dy(34) * (8.3026492595241645 * yz * (-3.0 * x2 + y2)); // 3*sqrt(385)*yz*(-3*x2 + y2)/(4*sqrt(pi))
    d.z += (float)dL_dy(34) * (-12.453973889286246 * x2 * y2 + 2.0756623148810411 * x4 +
                               2.0756623148810411 * y4); // 3*sqrt(385)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    d.x += (float)dL_dy(35) * (19.6914617052051 * x2 * y2 - 3.2819102842008503 * x4 -
                               3.2819102842008503 * y4);             // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    d.y += (float)dL_dy(35) * (13.127641136803401 * xy * (x2 - y2)); // 15*sqrt(154)*xy*(x2 - y2)/(8*sqrt(pi))
    // d.z += (float)dL_dy(35) * (0);                               // 0
    if (degree <= 6) {
        return d;
    }
    d.x +=
        (float)dL_dy(36) * (4.0991046311514854 * y *
                            (-10.0 * x2 * y2 + 5.0 * x4 + y4)); // 3*sqrt(6006)*y*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
    d.y +=
        (float)dL_dy(36) * (4.0991046311514854 * x *
                            (-10.0 * x2 * y2 + x4 + 5.0 * y4)); // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    // d.z += (float)dL_dy(36) * (0);                               // 0
    d.x += (float)dL_dy(37) * (47.332383244635047 * xy * z * (-x2 + y2)); // 15*sqrt(2002)*xy*z*(-x2 + y2)/(8*sqrt(pi))
    d.y += (float)dL_dy(37) *
           (11.833095811158762 * z * (6.0 * x2 * y2 - x4 - y4)); // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    d.z += (float)dL_dy(37) * (2.3666191622317521 * y *
                               (10.0 * x2 * y2 - 5.0 * x4 - y4)); // 3*sqrt(2002)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    d.x += (float)dL_dy(38) * (2.0182596029148963 * y * (3.0 * x2 - y2) *
                               (11.0 * z2 - 1.0)); // 3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    d.y += (float)dL_dy(38) * (2.0182596029148963 * x * (x2 - 3.0 * y2) *
                               (11.0 * z2 - 1.0)); // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
    d.z += (float)dL_dy(38) * (44.401711264127719 * xy * z * (x2 - y2)); // 33*sqrt(91)*xy*z*(x2 - y2)/(4*sqrt(pi))
    d.x += (float)dL_dy(39) *
           (5.5272315570895412 * xy * z * (3.0 - 11.0 * z2)); // 3*sqrt(2730)*xy*z*(3 - 11*z2)/(16*sqrt(pi))
    d.y += (float)dL_dy(39) * (-2.7636157785447706 * z * (x2 - y2) *
                               (11.0 * z2 - 3.0)); // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    d.z += (float)dL_dy(39) * (-2.7636157785447706 * y * (3.0 * x2 - y2) *
                               (11.0 * z2 - 1.0)); // -3*sqrt(2730)*y*(3*x2 - y2)*(11*z2 - 1)/(32*sqrt(pi))
    d.x += (float)dL_dy(40) * (0.92120525951492349 * y *
                               (-18.0 * z2 + 33.0 * z4 + 1.0)); // sqrt(2730)*y*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    d.y += (float)dL_dy(40) * (0.92120525951492349 * x *
                               (-18.0 * z2 + 33.0 * z4 + 1.0)); // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    d.z += (float)dL_dy(40) *
           (11.054463114179082 * xy * z * (11.0 * z2 - 3.0)); // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(8*sqrt(pi))
    // d.x += (float)dL_dy(41) * (0);                               // 0
    d.y += (float)dL_dy(41) *
           (0.58262136251873131 * z * (30.0 * z2 - 33.0 * z4 - 5.0)); // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    d.z += (float)dL_dy(41) *
           (2.9131068125936568 * y * (18.0 * z2 - 33.0 * z4 - 1.0)); // 5*sqrt(273)*y*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
    // d.x += (float)dL_dy(42) * (0);                               // 0
    // d.y += (float)dL_dy(42) * (0);                               // 0
    d.z += (float)dL_dy(42) * (2.6699064952403937 * z *
                               (-30.0 * z2 + 33.0 * z4 + 5.0)); // 21*sqrt(13)*z*(-30*z2 + 33*z4 + 5)/(16*sqrt(pi))
    d.x += (float)dL_dy(43) *
           (0.58262136251873131 * z * (30.0 * z2 - 33.0 * z4 - 5.0)); // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    // d.y += (float)dL_dy(43) * (0);                               // 0
    d.z += (float)dL_dy(43) *
           (2.9131068125936568 * x * (18.0 * z2 - 33.0 * z4 - 1.0)); // 5*sqrt(273)*x*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
    d.x += (float)dL_dy(44) * (0.92120525951492349 * x *
                               (-18.0 * z2 + 33.0 * z4 + 1.0)); // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    d.y += (float)dL_dy(44) *
           (0.92120525951492349 * y * (18.0 * z2 - 33.0 * z4 - 1.0)); // sqrt(2730)*y*(18*z2 - 33*z4 - 1)/(32*sqrt(pi))
    d.z += (float)dL_dy(44) * (5.5272315570895412 * z * (x2 - y2) *
                               (11.0 * z2 - 3.0)); // 3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(16*sqrt(pi))
    d.x += (float)dL_dy(45) * (-2.7636157785447706 * z * (x2 - y2) *
                               (11.0 * z2 - 3.0)); // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    d.y += (float)dL_dy(45) *
           (5.5272315570895412 * xy * z * (11.0 * z2 - 3.0)); // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(16*sqrt(pi))
    d.z += (float)dL_dy(45) * (-2.7636157785447706 * x * (x2 - 3.0 * y2) *
                               (11.0 * z2 - 1.0)); // -3*sqrt(2730)*x*(x2 - 3*y2)*(11*z2 - 1)/(32*sqrt(pi))
    d.x += (float)dL_dy(46) * (2.0182596029148963 * x * (x2 - 3.0 * y2) *
                               (11.0 * z2 - 1.0)); // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
    d.y += (float)dL_dy(46) * (-2.0182596029148963 * y * (3.0 * x2 - y2) *
                               (11.0 * z2 - 1.0)); // -3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    d.z += (float)dL_dy(46) *
           (11.10042781603193 * z * (-6.0 * x2 * y2 + x4 + y4)); // 33*sqrt(91)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    d.x += (float)dL_dy(47) *
           (11.833095811158762 * z * (6.0 * x2 * y2 - x4 - y4)); // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
    d.y += (float)dL_dy(47) * (47.332383244635047 * xy * z * (x2 - y2)); // 15*sqrt(2002)*xy*z*(x2 - y2)/(8*sqrt(pi))
    d.z += (float)dL_dy(47) * (2.3666191622317521 * x *
                               (10.0 * x2 * y2 - x4 - 5.0 * y4)); // 3*sqrt(2002)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    d.x +=
        (float)dL_dy(48) * (4.0991046311514854 * x *
                            (-10.0 * x2 * y2 + x4 + 5.0 * y4)); // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    d.y += (float)dL_dy(48) * (4.0991046311514854 * y *
                               (10.0 * x2 * y2 - 5.0 * x4 - y4)); // 3*sqrt(6006)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    // d.z += (float)dL_dy(48) * (0);                               // 0
    if (degree <= 7) {
        return d;
    }
    d.x += (float)dL_dy(49) *
           (9.9002782553443485 * xy *
            (10.0 * x2 * y2 - 3.0 * x4 - 3.0 * y4)); // 21*sqrt(715)*xy*(10*x2*y2 - 3*x4 - 3*y4)/(32*sqrt(pi))
    d.y += (float)dL_dy(49) * (-74.252086915082614 * x2 * y4 + 74.252086915082614 * x4 * y2 - 4.9501391276721742 * x6 +
                               4.9501391276721742 * y6); // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
    // d.z += (float)dL_dy(49) * (0);                               // 0
    d.x += (float)dL_dy(50) *
           (15.875763970811402 * yz *
            (-10.0 * x2 * y2 + 5.0 * x4 + y4)); // 9*sqrt(10010)*yz*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
    d.y += (float)dL_dy(50) *
           (15.875763970811402 * xz *
            (-10.0 * x2 * y2 + x4 + 5.0 * y4)); // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    d.z += (float)dL_dy(50) *
           (5.2919213236038001 * xy *
            (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)); // 3*sqrt(10010)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    d.x += (float)dL_dy(51) * (-10.378311574405206 * xy * (x2 - y2) *
                               (13.0 * z2 - 1.0)); // -15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
    d.y += (float)dL_dy(51) * (0.51891557872026028 * (13.0 * z2 - 1.0) *
                               (10.0 * x2 * y2 - 5.0 * x4 + 4.0 * y2 * (5.0 * x2 - y2) -
                                y4)); // 3*sqrt(385)*(13*z2 - 1)*(10*x2*y2 - 5*x4 + 4*y2*(5*x2 - y2)
                                      // - y4)/(64*sqrt(pi))
    d.z +=
        (float)dL_dy(51) * (13.491805046726766 * yz *
                            (10.0 * x2 * y2 - 5.0 * x4 - y4)); // 39*sqrt(385)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    d.x += (float)dL_dy(52) * (4.1513246297620823 * yz * (3.0 * x2 - y2) *
                               (13.0 * z2 - 3.0)); // 3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    d.y += (float)dL_dy(52) * (4.1513246297620823 * xz * (x2 - 3.0 * y2) *
                               (13.0 * z2 - 3.0)); // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
    d.z += (float)dL_dy(52) * (12.453973889286248 * xy * (x2 - y2) *
                               (13.0 * z2 - 1.0)); // 9*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(8*sqrt(pi))
    d.x += (float)dL_dy(53) * (0.93875360317376422 * xy *
                               (66.0 * z2 - 143.0 * z4 - 3.0)); // 9*sqrt(35)*xy*(66*z2 - 143*z4 - 3)/(32*sqrt(pi))
    d.y +=
        (float)dL_dy(53) * (-0.46937680158688211 * (x2 - y2) *
                            (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)); // -9*sqrt(35)*(x2 - y2)*(13*z2*(11*z2 -
                                                                                // 3) - 27*z2 + 3)/(64*sqrt(pi))
    d.z += (float)dL_dy(53) * (-6.8841930899409371 * yz * (3.0 * x2 - y2) *
                               (13.0 * z2 - 3.0)); // -33*sqrt(35)*yz*(3*x2 - y2)*(13*z2 - 3)/(16*sqrt(pi))
    d.x +=
        (float)dL_dy(54) * (0.44253269244498261 * yz *
                            (-110.0 * z2 + 143.0 * z4 + 15.0)); // 3*sqrt(70)*yz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    d.y +=
        (float)dL_dy(54) * (0.44253269244498261 * xz *
                            (-110.0 * z2 + 143.0 * z4 + 15.0)); // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    d.z += (float)dL_dy(54) * (2.2126634622249131 * xy *
                               (-66.0 * z2 + 143.0 * z4 + 3.0)); // 15*sqrt(70)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
    // d.x += (float)dL_dy(55) * (0);                               // 0
    d.y += (float)dL_dy(55) * (-12.194767023639836 * z2 + 44.714145753346067 * z4 - 38.752259652899923 * z6 +
                               0.45165803791258652); // sqrt(105)*(-135*z2 + 495*z4
                                                     // - 429*z6 + 5)/(64*sqrt(pi))
    d.z += (float)dL_dy(55) * (1.6259689364853116 * yz *
                               (110.0 * z2 - 143.0 * z4 - 15.0)); // 9*sqrt(105)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
    // d.x += (float)dL_dy(56) * (0);                               // 0
    // d.y += (float)dL_dy(56) * (0);                               // 0
    d.z += (float)dL_dy(56) * (64.528641681844675 * z2 - 236.60501950009714 * z4 + 205.05768356675085 * z6 -
                               2.3899496919201733); // 7*sqrt(15)*(135*z2 - 495*z4 +
                                                    // 429*z6 - 5)/(32*sqrt(pi))
    d.x += (float)dL_dy(57) * (-12.194767023639836 * z2 + 44.714145753346067 * z4 - 38.752259652899923 * z6 +
                               0.45165803791258652); // sqrt(105)*(-135*z2 + 495*z4
                                                     // - 429*z6 + 5)/(64*sqrt(pi))
    // d.y += (float)dL_dy(57) * (0);                               // 0
    d.z += (float)dL_dy(57) * (1.6259689364853116 * xz *
                               (110.0 * z2 - 143.0 * z4 - 15.0)); // 9*sqrt(105)*xz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
    d.x +=
        (float)dL_dy(58) * (0.44253269244498261 * xz *
                            (-110.0 * z2 + 143.0 * z4 + 15.0)); // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    d.y += (float)dL_dy(58) * (0.44253269244498261 * yz *
                               (110.0 * z2 - 143.0 * z4 - 15.0)); // 3*sqrt(70)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
    d.z += (float)dL_dy(58) * (0.07375544874083044 * (x2 - y2) *
                               (143.0 * z2 * (3.0 * z2 - 1.0) + 132.0 * z2 * (13.0 * z2 - 5.0) - 187.0 * z2 +
                                45.0)); // sqrt(70)*(x2 - y2)*(143*z2*(3*z2 - 1) +
                                        // 132*z2*(13*z2 - 5) - 187*z2 + 45)/(64*sqrt(pi))
    d.x += (float)dL_dy(59) * (30.97886890473422 * x2 * z2 - 67.120882626924143 * x2 * z4 - 1.4081304047606462 * x2 -
                               30.97886890473422 * y2 * z2 + 67.120882626924143 * y2 * z4 +
                               1.4081304047606462 * y2); // 9*sqrt(35)*(66*x2*z2 - 143*x2*z4 - 3*x2 -
                                                         // 66*y2*z2 + 143*y2*z4 + 3*y2)/(64*sqrt(pi))
    d.y += (float)dL_dy(59) * (0.93875360317376422 * xy *
                               (-66.0 * z2 + 143.0 * z4 + 3.0)); // 9*sqrt(35)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
    d.z += (float)dL_dy(59) * (-6.8841930899409371 * xz * (x2 - 3.0 * y2) *
                               (13.0 * z2 - 3.0)); // -33*sqrt(35)*xz*(x2 - 3*y2)*(13*z2 - 3)/(16*sqrt(pi))
    d.x += (float)dL_dy(60) * (4.1513246297620823 * xz * (x2 - 3.0 * y2) *
                               (13.0 * z2 - 3.0)); // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
    d.y += (float)dL_dy(60) * (-4.1513246297620823 * yz * (3.0 * x2 - y2) *
                               (13.0 * z2 - 3.0)); // -3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    d.z +=
        (float)dL_dy(60) * (3.1134934723215619 * (13.0 * z2 - 1.0) *
                            (-6.0 * x2 * y2 + x4 + y4)); // 9*sqrt(385)*(13*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    d.x +=
        (float)dL_dy(61) *
        (-0.51891557872026028 * (13.0 * z2 - 1.0) *
         (-10.0 * x2 * y2 + 4.0 * x2 * (x2 - 5.0 * y2) + x4 + 5.0 * y4)); // -3*sqrt(385)*(13*z2 - 1)*(-10*x2*y2 +
                                                                          // 4*x2*(x2 - 5*y2) + x4 + 5*y4)/(64*sqrt(pi))
    d.y += (float)dL_dy(61) * (10.378311574405206 * xy * (x2 - y2) *
                               (13.0 * z2 - 1.0)); // 15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
    d.z +=
        (float)dL_dy(61) * (13.491805046726766 * xz *
                            (10.0 * x2 * y2 - x4 - 5.0 * y4)); // 39*sqrt(385)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    d.x += (float)dL_dy(62) *
           (15.875763970811402 * xz *
            (-10.0 * x2 * y2 + x4 + 5.0 * y4)); // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
    d.y +=
        (float)dL_dy(62) * (15.875763970811402 * yz *
                            (10.0 * x2 * y2 - 5.0 * x4 - y4)); // 9*sqrt(10010)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    d.z += (float)dL_dy(62) * (39.6894099270285 * x2 * y4 - 39.6894099270285 * x4 * y2 + 2.6459606618019 * x6 -
                               2.6459606618019 * y6); // 3*sqrt(10010)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    d.x += (float)dL_dy(63) * (-74.252086915082614 * x2 * y4 + 74.252086915082614 * x4 * y2 - 4.9501391276721742 * x6 +
                               4.9501391276721742 * y6); // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
    d.y += (float)dL_dy(63) *
           (9.9002782553443485 * xy *
            (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)); // 21*sqrt(715)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    // d.z += (float)dL_dy(63) * (0);                               // 0
    return d;
}

template <uint32_t N_DIMS, uint32_t N_PRIMES>
uint32_t lcg_hash(const tnn::uvec<N_DIMS> &pos_grid, const uint32_t primes[N_PRIMES]) {
    static_assert(N_DIMS <= N_PRIMES, "lcg_hash can only hash up to N_PRIMES dimensions.");

    uint32_t result = 0;

    for (uint32_t i = 0; i < N_DIMS; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}

template <uint32_t N_DIMS> uint32_t prime_hash(const tnn::uvec<N_DIMS> &pos_grid) {
    constexpr uint32_t factors[7] = {1958374283u, 2654435761u, 805459861u, 3674653429u,
                                     2097192037u, 1434869437u, 2165219737u};
    return lcg_hash<N_DIMS, 7>(pos_grid, factors);
}

template <uint32_t N_DIMS> uint32_t coherent_prime_hash(const tnn::uvec<N_DIMS> &pos_grid) {
    constexpr uint32_t factors[7] = {1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u};
    return lcg_hash<N_DIMS, 7>(pos_grid, factors);
}

template <uint32_t N_DIMS> uint32_t reversed_prime_hash(const tnn::uvec<N_DIMS> &pos_grid) {
    constexpr uint32_t factors[7] = {2165219737u, 1434869437u, 2097192037u, 3674653429u,
                                     805459861u,  2654435761u, 1958374283u};
    return lcg_hash<N_DIMS, 7>(pos_grid, factors);
}

template <uint32_t N_DIMS, HashType HASH_TYPE> uint32_t grid_hash(const tnn::uvec<N_DIMS> &pos_grid) {
    switch (HASH_TYPE) {
    case HashType::Prime:
        return prime_hash<N_DIMS>(pos_grid);
    case HashType::CoherentPrime:
        return coherent_prime_hash<N_DIMS>(pos_grid);
    case HashType::ReversedPrime:
        return reversed_prime_hash<N_DIMS>(pos_grid);
    case HashType::Rng:
        return 0;
        // return rng_hash<N_DIMS>(pos_grid);
    }

    return 0;
}

template <uint32_t N_DIMS, HashType HASH_TYPE>
uint32_t grid_index(const GridType grid_type, const uint32_t hashmap_size, const uint32_t grid_resolution,
                    const tnn::uvec<N_DIMS> &pos_grid) {
    uint32_t stride = 1;
    uint32_t index = 0;

    // The second part of the loop condition is needed to avoid integer overflows
    // in finer levels.

    for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim) {
        index += pos_grid[dim] * stride;
        stride *= grid_resolution;
    }

    if (grid_type == GridType::Hash && hashmap_size < stride) {
        index = grid_hash<N_DIMS, HASH_TYPE>(pos_grid);
    }

    return index % hashmap_size;
}

inline float grid_scale(const uint32_t level, const float log2_per_level_scale, const uint32_t base_resolution) {
    // The -1 means that `base_resolution` refers to the number of grid _vertices_
    // rather than the number of cells. This is slightly different from the
    // notation in the paper, but results in nice, power-of-2-scaled parameter
    // grids that fit better into cache lines.
    return sycl::exp2(level * log2_per_level_scale) * base_resolution - 1.0f;
}

inline uint32_t grid_resolution(float scale) { return (uint32_t)sycl::ceil(scale) + 1; }

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline float smoothstep(const float val) { return val * val * (3.0f - 2.0f * val); }

inline float smoothstep_derivative(const float val) { return 6 * val * (1.0f - val); }

inline float smoothstep_2nd_derivative(const float val) { return 6.0f - 12.0f * val; }

inline float identity_fun(const float val) { return val; }

inline float identity_derivative(const float val) { return 1.0f; }

inline float identity_2nd_derivative(const float val) { return 0.0f; }

inline float quartic(const float x, const float inv_radius) {
    const float u = x * inv_radius;
    const float tmp = sycl::fmax(1 - u * u, 0.0f);
    return ((float)15 / 16) * tmp * tmp;
}

inline float quartic_cdf_deriv(const float x, const float inv_radius) { return quartic(x, inv_radius) * inv_radius; }

inline float quartic_cdf(const float x, const float inv_radius) {
    const float u = x * inv_radius;
    const float u2 = u * u;
    const float u4 = u2 * u2;
    return sycl::fmax(0.0f,
                      sycl::fmin(1.0f, ((float)15 / 16) * u * (1 - ((float)2 / 3) * u2 + ((float)1 / 5) * u4) + 0.5f));
}