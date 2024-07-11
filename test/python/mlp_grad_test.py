import numpy as np
import torch
from src.mlp import MLP

if torch.cuda.is_available():
    import tinycudann as tcnn

    device = "cuda"
else:
    import tiny_dpcpp_nn as tcnn

    device = "xpu"
torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)


to_numpy = lambda a: a.detach().cpu().numpy()

torch.manual_seed(42)


def is_close(reference, value, rtol=1e-4, name="", print_diff=False):
    assert len(reference.shape) == 1, "Reference should be a flat vector"
    assert len(value.shape) == 1, "Value should be a flat vector"

    max_val = np.maximum(np.max(np.abs(reference)), np.max(np.abs(value)))
    max_rtol = 0.0
    # Perform the element-wise comparison
    isclose = True
    for i, (a, b) in enumerate(zip(reference, value)):
        abs_diff = np.abs(a - b)
        rel_diff = abs_diff / max_val
        if rel_diff > max_rtol:
            max_rtol = rel_diff
        if rel_diff > rtol:
            isclose = False
            if print_diff:
                string = f" of {name}" if name else "name"
                print(f"Element {i}{string}:")
                print(f"  Value in reference (cuda): {a}")
                print(f"  Value in value (dpcpp): {b}")
                print(f"  Absolute difference: {abs_diff}")
                print(f"  Relative difference: {rel_diff} with rtol {rtol}")
    return isclose, max_rtol


def get_gradient_ref(input, weight_val, config, dtype=torch.float32):
    model = MLP(
        config["n_neurons"],
        [config["n_neurons"]] * config["n_hidden_layers"],
        config["n_neurons"],
        activation_func=config["activation"],
        output_activation=config["output_activation"],
        dtype=dtype,
        constant_weight=True,
        weight_val=weight_val,
    ).to(device)
    y_ref = model(input.to(device))
    y_ref.backward(torch.ones_like(y_ref))
    grads_all = []
    params_all = []

    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            gradient = param.grad

            grad = gradient.clone()
            grads_all.append(grad)

        params_all.append(param.data.clone())

    return torch.stack(grads_all).to("cpu"), torch.stack(params_all).to("cpu")


def run_config(config, weight_val, input_val):
    # avoid padding
    n_input_dims = config["n_neurons"]
    n_output_dims = config["n_neurons"]

    network = tcnn.Network(n_input_dims, n_output_dims, config)

    # tiny-cuda-nn seems to use always float32 for params
    assert network.params.dtype == torch.float32

    # make sure we have the same initialization
    torch.manual_seed(42)
    params = (
        torch.distributions.uniform.Uniform(-0.216, 0.216)
        .sample(network.params.shape)
        .to(device)
    )

    if weight_val != "random":
        params = params * 0 + weight_val

    if device == "cuda":
        network.params.data[...] = params
    else:
        network.set_params(params)

    x = (
        torch.distributions.uniform.Uniform(0.01, 1)
        .sample((1024, n_input_dims))
        .to(device)
    )

    if input_val != "random":
        x = x * 0 + input_val

    y = network(x)
    y.backward(torch.ones_like(y))
    weight_val_string = (
        f"1e{np.log10(weight_val):.0f}" if weight_val != "random" else "random"
    )
    input_val_string = (
        f"1e{np.log10(input_val):.0f}" if input_val != "random" else "random"
    )

    filename = f"output/mlp_grad_test_tensors_{config['activation']}_{config['output_activation']}_{config['n_neurons']}_{config['n_hidden_layers']}_weights{weight_val_string}_constant_input{input_val_string}.npz"
    print(
        f"Running {filename} with config {config}, weights: {weight_val_string}, input: {input_val_string}"
    )
    if device == "cuda":
        np.savez_compressed(
            filename,
            x=to_numpy(x),
            y=to_numpy(y),
            params=to_numpy(network.params),
            params_grad=to_numpy(network.params.grad),
        )
    elif device == "xpu":
        reference_grads_single_precision, reference_params_single_precision = (
            get_gradient_ref(x, weight_val, config, dtype=torch.float32)
        )
        reference_grads_half_precision, reference_params_half_precision = (
            get_gradient_ref(x, weight_val, config, dtype=torch.float16)
        )

        # Decide rtol_ref by comparing the rtol diff between reference in fp16 and fp32
        _, rtol_ref = is_close(
            to_numpy(reference_grads_single_precision).flatten(),
            to_numpy(reference_grads_half_precision).flatten(),
        )

        # Decide rtol_ref by comparing the rtol diff between reference in fp16 and fp32
        _, rtol_weights = is_close(
            np.array([reference_params_single_precision.sum()]),
            np.array([reference_params_half_precision.sum()]),
        )

        print(f"Using rtol_ref: {rtol_ref} and {rtol_weights} for weights")
        cuda = np.load(filename)
        x_isclose, _ = is_close(
            cuda["x"].flatten(), to_numpy(x).flatten(), name="Input", rtol=rtol_ref
        )
        assert x_isclose
        y_isclose, _ = is_close(
            cuda["y"].flatten(), to_numpy(y).flatten(), name="Output", rtol=rtol_ref
        )
        assert y_isclose

        params_isclose, _ = is_close(
            np.array([cuda["params"].sum()]),
            np.array([to_numpy(network.params.sum())]),
            name="Params",
            rtol=rtol_weights,
        )
        assert params_isclose

        grads_isclose, _ = is_close(
            cuda["params_grad"].flatten(),
            to_numpy(network.params.grad).flatten(),
            name="Grads",
            rtol=rtol_ref,
        )

        if not grads_isclose and weight_val != "random":

            dpcpp_grads_isclose, _ = is_close(
                to_numpy(network.params.grad).flatten(),
                to_numpy(reference_grads_single_precision).flatten(),
                name="Grads: dpcpp vs ref",
                print_diff=False,
                rtol=rtol_ref,  # this value is decided by comparing the rtol diff between reference in fp16 and fp32
            )
            assert dpcpp_grads_isclose

            cuda_grads_isclose, _ = is_close(
                cuda["params_grad"].flatten(),
                to_numpy(reference_grads_single_precision).flatten(),
                name="Grads: cuda vs ref",
                print_diff=False,
                rtol=rtol_ref,
            )
            assert (
                not cuda_grads_isclose
            ), "Cuda reference is close to reference, but dpcpp is not close to cuda reference. Manually check"


if __name__ == "__main__":

    activation_funcs = ["none"]
    hidden_layer_counts = [1]
    hidden_sizes = [16]
    input_vals = [0.1]
    params_vals = [0.1]
    # activation_funcs = ["relu", "none", "sigmoid"]
    # hidden_layer_counts = [1, 2, 4]
    # hidden_sizes = [16, 32, 64, 128]
    # input_vals = [10**exponent for exponent in range(-5, 5)] + ["random"]
    # params_vals = [10**exponent for exponent in range(-5, 5)] + ["random"]

    for output_fn in activation_funcs:
        for activation_fn in activation_funcs:
            for hidden_layer in hidden_layer_counts:
                for n_neurons in hidden_sizes:
                    for input_val in input_vals:
                        for params_val in params_vals:
                            config = {
                                "otype": "FullyFusedMLP",
                                "activation": activation_fn,
                                "output_activation": output_fn,
                                "n_neurons": n_neurons,
                                "n_hidden_layers": hidden_layer,
                                "device": device,
                            }
                            run_config(config, params_val, input_val)
