import numpy as np
import torch
import pytest
import tiny_dpcpp_nn as tcnn

# this doesn't work for padded as it's packed (will compare zero vals with non-zero if compare packed vs non-packed)
input_sizes = [16]
output_sizes = [16]
dtypes = [torch.float16, torch.bfloat16]

device = "xpu"


@pytest.mark.parametrize(
    "input_size, output_size, dtype",
    [
        (input_size, output_size, dtype)
        for input_size in input_sizes
        for output_size in output_sizes
        for dtype in dtypes
    ],
)
def test_set_params(input_size, output_size, dtype):
    torch.manual_seed(42)

    WIDTH = 16
    n_input_dims = input_size
    n_output_dims = output_size
    BATCH_SIZE = 8
    x = (torch.ones(BATCH_SIZE, WIDTH) * 0.1).to(device, dtype=dtype)

    config = {
        "otype": "FullyFusedMLP",
        "activation": "None",
        "output_activation": "None",
        "n_neurons": WIDTH,
        "n_hidden_layers": 3,
        "device": device,
    }

    network = tcnn.Network(
        n_input_dims,
        n_output_dims,
        config,
        input_dtype=dtype,
        backend_param_dtype=dtype,
    )

    val = 0.123
    param_vals = val * torch.ones(WIDTH * WIDTH * 4, 1, dtype=dtype).to(device)
    network.set_params(param_vals)

    # Using torch.isclose to compare param_vals with network.params and network.params.data
    is_close_params = torch.isclose(param_vals, network.params.to(dtype))
    is_close_params_data = torch.isclose(param_vals, network.params.data.to(dtype))

    assert (
        is_close_params.all()
    ), "network.params before setting and after are not the same"
    assert (
        is_close_params_data.all()
    ), "network.params.data before setting and after are not the same"

    assert not torch.isnan(x).all(), "x is nan"
    assert not torch.isinf(x).all(), "x is inf"
    y = network(x)
    assert not torch.isnan(y).all(), "y is nan"
    assert not torch.isinf(y).all(), "y is inf"


if __name__ == "__main__":
    dtype = torch.float16
    input_size = 16
    output_size = 16
    test_set_params(input_size, output_size, dtype)
