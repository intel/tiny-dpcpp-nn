import numpy as np

from src.mlp import MLP

from tiny_dpcpp_nn import Network, NetworkWithInputEncoding
import torch


def is_close(reference, value, rtol=1e-4, name="", print_diff=False):
    assert len(reference.shape) == 1, "Reference should be a flat vector"
    assert len(value.shape) == 1, "Value should be a flat vector"

    max_val = np.maximum(np.max(np.abs(reference)), np.max(np.abs(value)))
    max_rtol = 0.0
    max_atol = 5e-4
    # Perform the element-wise comparison
    isclose = True
    for i, (a, b) in enumerate(zip(reference, value)):
        abs_diff = np.abs(a - b)

        rel_diff = abs_diff / max_val
        if rel_diff > max_rtol:
            max_rtol = rel_diff
        if rel_diff > rtol and abs_diff > max_atol:
            isclose = False
            if print_diff:
                string = f" of {name}" if name else ""
                print(f"Element {i}{string}:")
                print(f"  Value in reference (cuda): {a}")
                print(f"  Value in value (dpcpp): {b}")
                print(f"  Absolute difference: {abs_diff}")
                print(f"  Relative difference: {rel_diff} with rtol {rtol}")
    return isclose, max_rtol


def to_packed_layout_coord(idx, rows, cols):
    assert idx < rows * cols
    i = idx // cols
    j = idx % cols

    if i % 2 == 0:
        return i * cols + 2 * j
    else:
        return (i - 1) * cols + 2 * j + 1


def vertical_pack(matrix):
    rows, cols = matrix.shape
    packed = [0] * (rows * cols)  # Preallocate the packed array

    for idx in range(rows * cols):
        packed_idx = to_packed_layout_coord(idx, rows, cols)
        packed[packed_idx] = matrix.flatten()[idx]  # Use flat for 1D indexing

    return torch.tensor(packed).reshape(rows, cols)


def vertical_unpack(packed_matrix):
    rows, cols = packed_matrix.shape
    original = [0] * (rows * cols)  # Preallocate the original array

    for idx in range(rows * cols):
        packed_idx = to_packed_layout_coord(idx, rows, cols)
        original[idx] = packed_matrix.flatten()[packed_idx]  # Use flat for 1D indexing

    return torch.tensor(original).reshape(rows, cols)


def get_reshaped_params(
    weights,
    n_input_dims,
    width,
    n_output_dims,
    n_hidden_layers,
    dtype,
    device,
    mode,  # reshape, pack, unpack
    transpose_before_operation=True,  # this is necessary to convert from torch to dpcpp
):
    assert (
        len(weights.shape) == 1 or weights.shape[1] == 1
    ), "Weights is assumed to be a 1-D vector"

    input_matrix = weights[: width * width].reshape(width, width).to(dtype).to(device)

    len_input_matrix = input_matrix.shape[0] * input_matrix.shape[1]
    hidden_layer_size = width * width
    hidden_matrices = []

    for nth_hidden in range(n_hidden_layers - 1):
        hidden_matrix = (
            weights[
                len_input_matrix
                + nth_hidden * hidden_layer_size : len_input_matrix
                + (1 + nth_hidden) * hidden_layer_size
            ]
            .reshape(width, width)
            .to(dtype)
            .to(device)
        )

        hidden_matrices.append(hidden_matrix)

    output_matrix = weights[-width * width :].reshape(width, width).to(dtype).to(device)

    all_weights = []

    all_weights.append(input_matrix)
    all_weights.extend(hidden_matrices)
    all_weights.append(output_matrix)

    all_weights_changed = []
    for idx, layer in enumerate(all_weights):
        if mode == "pack":
            layer = vertical_pack(layer)
        elif mode == "unpack":
            layer = vertical_unpack(layer)

        layer_append = (
            layer.T.to(device) if transpose_before_operation else layer.to(device)
        )

        if idx == (len(all_weights) - 1):
            layer_append = layer_append[:n_output_dims, :]
        all_weights_changed.append(layer_append)
    return all_weights_changed


def get_unpacked_params(model, weights):
    return get_reshaped_params(
        weights,
        model.n_input_dims,
        model.width,
        model.n_output_dims,
        model.n_hidden_layers,
        model.dtype,
        model.device,
        "unpack",
    )


def get_grad_params(model):
    # This function unpacks for comparison with torch
    grads_all = []
    params_all = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            gradient = param.grad.clone()
            if len(gradient.shape) == 1 or param.data.shape[1] == 1:
                # for tiny-dpcpp-nn, need to unpack
                gradient = get_unpacked_params(model, gradient)

            grads_all.append(gradient)

        param_data = param.data.clone()
        if len(param_data.shape) == 1 or param_data.shape[1] == 1:
            # for tiny-dpcpp-nn, need to unpack
            param_data = get_unpacked_params(model, param_data)
        params_all.append(param_data)
    return grads_all, params_all


def compare_matrices(weights_dpcpp, weights_torch, rtol=1e-2):
    for layer, _ in enumerate(weights_dpcpp):
        assert (
            weights_dpcpp[layer].shape == weights_torch[layer].shape
        ), f"Shape different: dpcpp {weights_dpcpp[layer].shape} and torch {weights_torch[layer].shape}"
        are_close, _ = is_close(
            weights_dpcpp[layer].to(dtype=torch.float).flatten().to("cpu").numpy(),
            weights_torch[layer].to(dtype=torch.float).flatten().to("cpu").numpy(),
            rtol=rtol,
            name="",
            print_diff=True,
        )
        if not are_close:
            print(f"weights_dpcpp: {weights_dpcpp[layer]}")
            print(f"weights_torch: {weights_torch[layer]}")
            print(f"weights_dpcpp[{layer}] sum: {weights_dpcpp[layer].sum().sum()}")
            print(f"weights_torch[{layer}] sum: {weights_torch[layer].sum().sum()}")
        assert are_close


def create_models(
    input_size,
    hidden_sizes,
    output_size,
    activation_func,
    output_func,
    dtype,
    use_nwe,
    use_weights_of_tinynn,
    use_constant_weight=False,
    store_params_as_full_precision=False,
):
    # Create and test CustomMLP
    model_torch = MLP(
        input_size,
        hidden_sizes,
        output_size,
        activation_func,
        output_func,
        dtype=dtype,
        nwe_as_ref=use_nwe,
        constant_weight=use_constant_weight,
    )

    network_config = {
        "activation": activation_func,
        "output_activation": output_func,
        "n_neurons": hidden_sizes[0],
        "n_hidden_layers": len(hidden_sizes),
    }

    if use_nwe:
        encoding_config = {
            "otype": "Identity",
            "n_dims_to_encode": input_size,  # assuming the input size is 2 as in other tests
            "scale": 1.0,
            "offset": 0.0,
        }

        model_dpcpp = NetworkWithInputEncoding(
            n_input_dims=input_size,
            n_output_dims=output_size,
            encoding_config=encoding_config,
            network_config=network_config,
            store_params_as_full_precision=store_params_as_full_precision,
            dtype=dtype,
            use_bias=False,  # for comparison, we don't use the one padding
        )
    else:
        model_dpcpp = Network(
            n_input_dims=input_size,
            n_output_dims=output_size,
            network_config=network_config,
            store_params_as_full_precision=store_params_as_full_precision,
            dtype=dtype,
            use_bias=False,  # for comparison, we don't use the one padding
        )

    if use_weights_of_tinynn:
        weights = get_unpacked_params(model_dpcpp, model_dpcpp.params)
        model_torch.set_weights(weights)
    else:
        weights = model_torch.get_all_weights()
        model_dpcpp.set_params(weights.flatten())
    model_torch.to(model_dpcpp.device)
    return model_dpcpp, model_torch
