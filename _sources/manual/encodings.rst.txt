.. Copyright (C) 2024 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

===========
Encodings
===========

This documentation provides an overview of different encoding strategies, including Grid Encoding, Identity Encoding, and Spherical Harmonics Encoding. Please note that other Encodings are currently not supported.

Identity Encoding
=================

Identity Encoding is a straightforward method that retains the original value of each input dimension, with an optional scale and offset adjustment. This encoding is advantageous when inputs do not necessitate alteration.

JSON Configuration for Identity Encoding::

.. code-block:: json

    {
        "otype": "Identity",
        "scale": 1.0,
        "offset": 0.0
    }

Spherical Harmonics Encoding
============================

Spherical Harmonics Encoding optimizes 3D direction vectors, normalizing them into the unit cube. This frequency-space encoding offers a more efficient means of representing 3D data compared to component-wise encodings like Frequency or TriangleWave, particularly in capturing directional information.

JSON Configuration for Spherical Harmonics Encoding::

.. code-block:: json

    {
        "otype": "SphericalHarmonics",
        "degree": 4
    }

Grid Encoding
=============

Grid Encoding employs a trainable, multiresolution approach suitable for Instant Neural Graphics Primitives (`Müller et al. 2022 <https://nvlabs.github.io/instant-ngp/>`_). It allows for several backing storage methods: hashtable-backed (Hash), densely stored (Dense), or tiled storage (Tiled). The encoding's dimensionality is a product of the number of levels and features per level. Interpolation methods like Nearest, Linear, or Smoothstep determine the blending of neighboring features.

JSON Configuration for Grid Encoding::

.. code-block:: json

    {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 2.0,
        "interpolation": "Linear"
    }