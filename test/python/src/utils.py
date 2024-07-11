import numpy as np

from src.mlp import MLP

from tiny_dpcpp_nn import Network, NetworkWithInputEncoding
import torch


def get_grad_params(model):
    # This funciton unpacks for comparison with torch
    grads_all = []
    params_all = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            gradient = param.grad
            if len(gradient.shape) == 1 or param.data.shape[1] == 1:
                grad = model.get_reshaped_params(
                    gradient.clone(), is_packed_format=True
                )
            else:
                grad = gradient.clone()
            grads_all.append(grad)

        if len(param.data.shape) == 1 or param.data.shape[1] == 1:
            param_reshaped = model.get_reshaped_params(
                param.data.clone(), is_packed_format=True
            )
        else:
            param_reshaped = param.data.clone()

        params_all.append(param_reshaped)
    return grads_all, params_all


def compare_matrices(weights_dpcpp, weights_torch, atol=1e-1, rtol=5e-2):
    for layer, _ in enumerate(weights_dpcpp):
        assert (
            weights_dpcpp[layer].shape == weights_torch[layer].shape
        ), f"Shape different: {weights_dpcpp[layer].shape} x {weights_torch[layer].shape}"

        are_close = torch.allclose(
            weights_dpcpp[layer].to(dtype=torch.float),
            weights_torch[layer].to(dtype=torch.float),
            atol=atol,
        ) or torch.allclose(
            weights_dpcpp[layer].to(dtype=torch.float),
            weights_torch[layer].to(dtype=torch.float),
            rtol=rtol,
        )
        if not are_close:
            print(f"weights_dpcpp: {weights_dpcpp}")
            print(f"weights_torch: {weights_torch}")
            print(f"weights_dpcpp[layer] sum: {weights_dpcpp[layer].sum().sum()}")
            print(f"weights_torch[layer] sum: {weights_torch[layer].sum().sum()}")
        assert are_close


def create_models(
    input_size,
    hidden_sizes,
    output_size,
    activation_func,
    output_func,
    input_dtype,
    backend_param_dtype,
    use_nwe,
    use_weights_of_tinynn,
    constant_weight=False,
):

    # Create and test CustomMLP
    model_torch = MLP(
        input_size,
        hidden_sizes,
        output_size,
        activation_func,
        output_func,
        dtype=backend_param_dtype,
        nwe_as_ref=use_nwe,
        constant_weight=constant_weight,
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
            input_dtype=input_dtype,
            backend_param_dtype=backend_param_dtype,
        )
    else:
        model_dpcpp = Network(
            n_input_dims=input_size,
            n_output_dims=output_size,
            network_config=network_config,
            input_dtype=input_dtype,
            backend_param_dtype=backend_param_dtype,
        )

    if use_weights_of_tinynn:
        weights = model_dpcpp.get_reshaped_params()
        model_torch.set_weights(weights)
    else:
        weights = model_torch.get_all_weights()
        model_dpcpp.set_params(weights.flatten())

    model_torch.to(model_dpcpp.device)
    return model_dpcpp, model_torch
