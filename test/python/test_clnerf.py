import pytest
import numpy as np

from tiny_dpcpp_nn import Encoding, Network, NetworkWithInputEncoding

dtypes = [torch.float16, torch.bfloat16]

@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in dtypes],
)
test_constructor(dtype):
    scale = 1
    N_min = 16
    L = 16

    xyz_encoder = NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=16,
        encoding_config={
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": N_min,
            "per_level_scale": np.exp(np.log(2048 * scale / N_min) / (L - 1)),
            "interpolation": "Linear",
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 1,
        },
        dtype=dtype
    )

    rgb_net = Network(
        n_input_dims=32,
        n_output_dims=3,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "sigmoid",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
        dtype=dtype
    )

    tonemapper_net = Network(
        n_input_dims=1,
        n_output_dims=1,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": 64,
            "n_hidden_layers": 1,
        },
        dtype=dtype
    )

    sigma_mlp = Network(
        n_input_dims=64,
        n_output_dims=16,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 1,
        },
        dtype=dtype
    )

    dir_encoder = Encoding(
        n_input_dims=3,
        encoding_config={
            "otype": "SphericalHarmonics",
            "degree": 4,
        },
    )
