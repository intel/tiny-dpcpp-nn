/**
 * @file doctest_swiftnet.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the Swiftnet class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"

#include "common.h"
#include "result_check.h"

TEST_CASE("Testing get_packed_weights for 16") {

    // Reference values for comparison
    std::vector<float> reference_values = {
        1,   17,  2,   18,  3,   19,  4,   20,  5,   21,  6,   22,  7,   23,  8,   24,  9,   25,  10,  26,  11,  27,
        12,  28,  13,  29,  14,  30,  15,  31,  16,  32,  33,  49,  34,  50,  35,  51,  36,  52,  37,  53,  38,  54,
        39,  55,  40,  56,  41,  57,  42,  58,  43,  59,  44,  60,  45,  61,  46,  62,  47,  63,  48,  64,  65,  81,
        66,  82,  67,  83,  68,  84,  69,  85,  70,  86,  71,  87,  72,  88,  73,  89,  74,  90,  75,  91,  76,  92,
        77,  93,  78,  94,  79,  95,  80,  96,  97,  113, 98,  114, 99,  115, 100, 116, 101, 117, 102, 118, 103, 119,
        104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 112, 128, 129, 145, 130, 146,
        131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157,
        142, 158, 143, 159, 144, 160, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184,
        169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 176, 192, 193, 209, 194, 210, 195, 211,
        196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222,
        207, 223, 208, 224, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249,
        234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255, 240, 256, 257, 273, 258, 274, 259, 275, 260, 276,
        261, 277, 262, 278, 263, 279, 264, 280, 265, 281, 266, 282, 267, 283, 268, 284, 269, 285, 270, 286, 271, 287,
        272, 288, 289, 305, 290, 306, 291, 307, 292, 308, 293, 309, 294, 310, 295, 311, 296, 312, 297, 313, 298, 314,
        299, 315, 300, 316, 301, 317, 302, 318, 303, 319, 304, 320, 321, 337, 322, 338, 323, 339, 324, 340, 325, 341,
        326, 342, 327, 343, 328, 344, 329, 345, 330, 346, 331, 347, 332, 348, 333, 349, 334, 350, 335, 351, 336, 352,
        353, 369, 354, 370, 355, 371, 356, 372, 357, 373, 358, 374, 359, 375, 360, 376, 361, 377, 362, 378, 363, 379,
        364, 380, 365, 381, 366, 382, 367, 383, 368, 384, 385, 401, 386, 402, 387, 403, 388, 404, 389, 405, 390, 406,
        391, 407, 392, 408, 393, 409, 394, 410, 395, 411, 396, 412, 397, 413, 398, 414, 399, 415, 400, 416, 417, 433,
        418, 434, 419, 435, 420, 436, 421, 437, 422, 438, 423, 439, 424, 440, 425, 441, 426, 442, 427, 443, 428, 444,
        429, 445, 430, 446, 431, 447, 432, 448, 449, 465, 450, 466, 451, 467, 452, 468, 453, 469, 454, 470, 455, 471,
        456, 472, 457, 473, 458, 474, 459, 475, 460, 476, 461, 477, 462, 478, 463, 479, 464, 480, 481, 497, 482, 498,
        483, 499, 484, 500, 485, 501, 486, 502, 487, 503, 488, 504, 489, 505, 490, 506, 491, 507, 492, 508, 493, 509,
        494, 510, 495, 511, 496, 512,
    };

    int input_width = 16;
    int network_width = 16;
    int output_width = 16;
    int m_n_hidden_layers = 1; // Adjust as needed for testing

    // Create a vector with values from 1 to 16 * (16 + 16) for testing
    std::vector<float> unpacked_weights;
    for (int i = 1; i <= input_width * network_width + (m_n_hidden_layers - 1) * network_width * network_width +
                             network_width * output_width;
         ++i) {
        unpacked_weights.push_back(static_cast<float>(i));
    }

    auto packed_weights =
        get_packed_weights(unpacked_weights, m_n_hidden_layers, input_width, network_width, output_width);

    // Check the size of the packed weights
    CHECK(packed_weights.size() == reference_values.size());

    // Check the shape (this part may depend on your shape representation)
    CHECK(packed_weights.size() % network_width == 0); // Should be multiple of network_width

    // Check packed values against reference
    for (size_t i = 0; i < packed_weights.size(); ++i) {
        CHECK(packed_weights[i] == reference_values[i]);
    }
}
