/**
 * @file encoding_factory.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Factory class to generate the various encodings.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <string>

#include "encoding.h"
#include "grid.h"
#include "identity.h"
#include "spherical_harmonics.h"

// Base EncodingFactory class
template <typename T> class EncodingFactory {
  public:
    virtual ~EncodingFactory() {}
    virtual std::shared_ptr<Encoding<T>> create(const json &config) const = 0;
};

// EncodingFactory for IdentityEncoding
template <typename T> class IdentityEncodingFactory : public EncodingFactory<T> {
  public:
    std::shared_ptr<Encoding<T>> create(const json &config) const override {

        if (!config.contains(EncodingParams::OFFSET))
            throw std::invalid_argument("Config misses " + EncodingParams::OFFSET);
        if (!config.contains(EncodingParams::SCALE))
            throw std::invalid_argument("Config misses " + EncodingParams::SCALE);
        if (!config.contains(EncodingParams::N_DIMS_TO_ENCODE))
            throw std::invalid_argument("Config misses " + EncodingParams::N_DIMS_TO_ENCODE);

        const uint32_t n_dims_to_encode = config[EncodingParams::N_DIMS_TO_ENCODE];
        const float scale = config[EncodingParams::SCALE];
        const float offset = config[EncodingParams::OFFSET];
        return std::make_shared<IdentityEncoding<T>>(n_dims_to_encode, scale, offset);
    }
};

// EncodingFactory for SphericalHarmonicsEncoding
template <typename T> class SphericalHarmonicsEncodingFactory : public EncodingFactory<T> {
  public:
    std::shared_ptr<Encoding<T>> create(const json &config) const override {

        if (!config.contains(EncodingParams::DEGREE))
            throw std::invalid_argument("Config misses " + EncodingParams::DEGREE);
        if (!config.contains(EncodingParams::N_DIMS_TO_ENCODE))
            throw std::invalid_argument("Config misses " + EncodingParams::N_DIMS_TO_ENCODE);

        const uint32_t degree = config[EncodingParams::DEGREE];
        const uint32_t n_dims_to_encode = config[EncodingParams::N_DIMS_TO_ENCODE];
        return std::make_shared<SphericalHarmonicsEncoding<T>>(degree, n_dims_to_encode);
    }
};

template <typename T> class GridEncodingFactory;

// Specialization for T = bf16 (exclude implementation)
template <>
class GridEncodingFactory<sycl::ext::oneapi::bfloat16> : public EncodingFactory<sycl::ext::oneapi::bfloat16> {
  public:
    std::shared_ptr<Encoding<sycl::ext::oneapi::bfloat16>> create(const json &config) const override {
        // Throw an error or handle the unsupported case for bf16
        throw std::runtime_error("GridEncodingFactory does not support bf16");
    }
};

// Specialization for T != bf16 (include implementation)
// EncodingFactory for GridEncodingTemplated
template <typename T> class GridEncodingFactory : public EncodingFactory<T> {
  public:
    std::shared_ptr<Encoding<T>> create(const json &config) const override {

        return tinydpcppnn::encodings::grid::create_grid_encoding<T>(config);
    }
};

// Create a map to associate encoding names with their factories
template <typename T> class EncodingFactoryRegistry {
  public:
    bool contains(const std::string &name) const { return factories_.count(name) > 0; }

    void registerFactory(const std::string &name, std::unique_ptr<EncodingFactory<T>> factory) {
        factories_[name] = std::move(factory);
    }

    std::shared_ptr<Encoding<T>> create(const json &config) const {
        std::string encoding_name = config[EncodingParams::ENCODING];
        if (!contains(encoding_name)) throw std::invalid_argument("Unknown encoding type: " + encoding_name);
        return factories_.at(encoding_name)->create(config);
    }

  private:
    std::unordered_map<std::string, std::unique_ptr<EncodingFactory<T>>> factories_;
};

void check_validity_of_config(const json &config) {
    if( !config.contains(EncodingParams::CHECK_CONFIG) || config[EncodingParams::CHECK_CONFIG] == false ) {
        return;
    }
    // Define a list of all possible encoding parameters, see encoding.h
    const std::vector<std::string> valid_keys = {
        EncodingParams::ENCODING,
        EncodingParams::N_DIMS_TO_ENCODE,
        EncodingParams::GRID_TYPE,
        EncodingParams::N_LEVELS,
        EncodingParams::N_FEATURES,
        EncodingParams::N_FEATURES_PER_LEVEL,
        EncodingParams::LOG2_HASHMAP_SIZE,
        EncodingParams::BASE_RESOLUTION,
        EncodingParams::PER_LEVEL_SCALE,
        EncodingParams::DEGREE,
        EncodingParams::SCALE,
        EncodingParams::OFFSET,
        EncodingParams::HASH,
        EncodingParams::INTERPOLATION_METHOD,
        EncodingParams::USE_STOCHASTIC_INTERPOLATION,
        EncodingParams::CHECK_CONFIG,
    };

    // Check if every key in the config is within the valid keys
    for (const auto &config_key : config.items()) {
        if (std::find(valid_keys.begin(), valid_keys.end(), config_key.key()) == valid_keys.end()) {
            // If a key in the config JSON is not found in the valid_keys list, throw an exception
            std::string supported_keys = "";
            for (const auto &key : valid_keys) {
                supported_keys += key + ", ";
            }
            if (!supported_keys.empty()) {
                supported_keys.pop_back(); // Remove trailing comma and space
                supported_keys.pop_back();
            }
            throw std::invalid_argument("Unsupported key found in config: " + config_key.key() +
                                        ". Supported keys are: [" + supported_keys + "]");
        }
    }
}

template <typename T> std::shared_ptr<Encoding<T>> create_encoding(const json &config) {
    check_validity_of_config(config);

    // Create a registry for encoding factories
    EncodingFactoryRegistry<T> encodingRegistry;
    if (!config.contains(EncodingParams::ENCODING))
        throw std::invalid_argument("Config needs to contain encoding, key " + EncodingParams::ENCODING);
    const std::string name = config[EncodingParams::ENCODING];

    if (!encodingRegistry.contains(name)) {
        if (name == EncodingNames::IDENTITY)
            encodingRegistry.registerFactory(name, std::make_unique<IdentityEncodingFactory<T>>());
        else if (name == EncodingNames::SPHERICALHARMONICS)
            encodingRegistry.registerFactory(name, std::make_unique<SphericalHarmonicsEncodingFactory<T>>());
        else if (name == EncodingNames::GRID)
            encodingRegistry.registerFactory(name, std::make_unique<GridEncodingFactory<T>>());
        else
            throw std::invalid_argument("Encoding name unknown: " + name +
                                        ". Currently supported: " + EncodingNames::IDENTITY + ", " +
                                        EncodingNames::SPHERICALHARMONICS + ", " + EncodingNames::GRID + ".");
    }

    return encodingRegistry.create(config);
}
