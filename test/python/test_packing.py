from src.utils import vertical_pack, vertical_unpack, get_reshaped_params
import numpy as np
import pytest
import torch
from tiny_dpcpp_nn import Network, NetworkWithInputEncoding
import random


def test_vertical_pack():
    A = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    packed_A = vertical_pack(A)
    expected_packed_A = np.array(
        [
            [1, 5, 2, 6],
            [3, 7, 4, 8],
            [9, 13, 10, 14],
            [11, 15, 12, 16],
        ]
    )
    np.testing.assert_array_equal(packed_A, expected_packed_A)


def test_vertical_unpack():
    packed_A = np.array(
        [
            [1, 5, 2, 6],
            [3, 7, 4, 8],
            [9, 13, 10, 14],
            [11, 15, 12, 16],
        ]
    )
    unpacked_A = vertical_unpack(packed_A)
    expected_unpacked_A = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    np.testing.assert_array_equal(unpacked_A, expected_unpacked_A)


def test_pack_unpack():
    A = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    packed_A = vertical_pack(A)
    unpacked_A = vertical_unpack(packed_A)
    np.testing.assert_array_equal(A, unpacked_A)


def test_16x16_pack_unpack():
    A = np.arange(1, 257).reshape(16, 16)
    packed_A = vertical_pack(A)
    unpacked_A = vertical_unpack(packed_A)
    np.testing.assert_array_equal(A, unpacked_A)


def test_16x16_unpack_pack():
    packed_A = vertical_pack(np.arange(1, 257).reshape(16, 16))
    unpacked_A = vertical_unpack(packed_A)
    repacked_A = vertical_pack(unpacked_A)
    np.testing.assert_array_equal(packed_A, repacked_A)


# Helper function to convert numpy array to torch tensor with specified dtype and device
def to_tensor(array, dtype, device):
    return torch.tensor(array, dtype=dtype).to(device)


# Test function
@pytest.mark.parametrize("width", [16, 32, 64, 128])
@pytest.mark.parametrize("n_hidden_layers", [1, 2, 3])
@pytest.mark.parametrize("mode", ["reshape", "pack", "unpack"])
def test_reshaped_params_by_shape(width, n_hidden_layers, mode):
    total_elements = (
        (width * width) * (n_hidden_layers - 1) + width * width + width * width
    )
    weights = torch.arange(1, total_elements + 1).float()

    dtype = torch.float32
    device = torch.device("cpu")

    reshaped_params = get_reshaped_params(
        weights, width, width, width, n_hidden_layers, dtype, device, mode, False
    )
    if mode == "reshape":
        assert len(reshaped_params) == n_hidden_layers + 1
        assert reshaped_params[0].shape == (width, width)
        for hidden_layer in reshaped_params[1:-1]:
            assert hidden_layer.shape == (width, width)
        assert reshaped_params[-1].shape == (width, width)
    elif mode == "pack":
        assert len(reshaped_params) == n_hidden_layers + 1
        assert reshaped_params[0].shape == (width, width)
        for hidden_layer in reshaped_params[1:-1]:
            assert hidden_layer.shape == (width, width)
        assert reshaped_params[-1].shape == (width, width)

    elif mode == "unpack":
        assert len(reshaped_params) == n_hidden_layers + 1
        assert reshaped_params[0].shape == (width, width)
        for hidden_layer in reshaped_params[1:-1]:
            assert hidden_layer.shape == (width, width)
        assert reshaped_params[-1].shape == (width, width)


# Test function
@pytest.mark.parametrize("width", [16, 32, 64, 128])
@pytest.mark.parametrize("n_hidden_layers", [1, 2, 3])
@pytest.mark.parametrize("mode", ["reshape", "pack", "unpack"])
def test_reshaped_params_by_values(width, n_hidden_layers, mode):
    total_elements = (
        (width * width) * (n_hidden_layers - 1) + width * width + width * width
    )
    weights = torch.arange(1, total_elements + 1).float()

    dtype = torch.float32
    device = torch.device("cpu")
    if mode == "unpack":
        reshaped_params = get_reshaped_params(
            weights, width, width, width, n_hidden_layers, dtype, device, "pack", False
        )  # need to pack first
        reshaped_params = get_reshaped_params(
            torch.stack(reshaped_params).flatten().squeeze(),
            width,
            width,
            width,
            n_hidden_layers,
            dtype,
            device,
            mode,
            False,
        )
    else:
        reshaped_params = get_reshaped_params(
            weights, width, width, width, n_hidden_layers, dtype, device, mode, False
        )

    assert len(reshaped_params) == n_hidden_layers + 1
    for i, layer in enumerate(reshaped_params):
        assert layer.shape == (width, width)
        start_idx = i * width * width
        end_idx = start_idx + width * width
        reference_matrix = torch.arange(1 + start_idx, 1 + end_idx).reshape(
            width, width
        )
        if mode == "reshape":
            np.testing.assert_array_equal(layer, reference_matrix)
        elif mode == "pack":
            expected_values = vertical_pack(reference_matrix)
            np.testing.assert_array_equal(layer, expected_values)
        elif mode == "unpack":
            np.testing.assert_array_equal(layer, reference_matrix)


if __name__ == "__main__":
    # test_vertical_pack()
    # test_vertical_unpack()
    # test_16x16_pack_unpack()
    # test_16x16_unpack_pack()
    test_reshaped_params_by_values(32, 2, "unpack")
    # test_reshaped_params_by_shape(16, 2, "reshape")
