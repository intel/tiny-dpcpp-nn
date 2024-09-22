import numpy as np
import torch
import pytest
from src.utils import get_reshaped_params

if torch.cuda.is_available():
    import tinycudann as tcnn

    device = "cuda"
else:
    import tiny_dpcpp_nn as tcnn

    device = "xpu"

to_numpy = lambda a: a.detach().cpu().numpy()


def plot_params(net, tensor=None):
    import cv2

    if tensor is None:
        arr = to_numpy(net.params.data)
    else:
        arr = to_numpy(tensor)
    arr = arr.reshape(-1, net.network_config["n_neurons"])
    arr = np.abs(arr)
    # arr = (arr > 0).astype(np.float32)
    arr = np.clip(255 * arr / arr.max(), 0, 255).astype(np.uint8)
    cv2.imwrite("/output/params.png", arr)
    print(arr.shape)


def round_up_mod(x, k: int = 16):
    if x % k:
        return (x // k + 1) * k
    return x


def cuda_params_to_numpy(network, params=None):
    n_input_dims = network.n_input_dims
    n_output_dims = network.n_output_dims
    n_neurons = network.network_config["n_neurons"]
    n_hidden_layers = network.network_config["n_hidden_layers"]

    if params is None:
        params = to_numpy(network.params.data)
    Ws = []
    pos = 0
    for layer in range(n_hidden_layers + 1):
        if layer == 0:
            rows = round_up_mod(n_input_dims, 16)
            cols = n_neurons
            size = rows * cols
        elif layer == n_hidden_layers:
            rows = n_neurons
            cols = n_output_dims
            size = rows * cols
        else:
            rows = n_neurons
            cols = n_neurons
            size = rows * cols

        Ws.append(params[pos : pos + size].reshape(cols, rows).T)
        pos += size

    def net_fn(x):
        # includes the padding with ones that tiny-cuda is doing
        x_pad = np.ones((1, round_up_mod(n_input_dims, 16)))
        x_pad[:, :n_input_dims] = x
        y = x_pad
        for W in Ws:
            y = y @ W
        return y

    return Ws, net_fn


def numpy_params_to_xpu(Ws, align):
    n_input_dims = Ws[0].shape[0]
    n_output_dims = Ws[-1].shape[1]
    n_neurons = Ws[-1].shape[0]

    size = 0
    for W in Ws:
        size += round_up_mod(W.shape[0], align) * round_up_mod(W.shape[1], align)

    params = np.zeros(shape=(1, size))
    pos = 0
    for W in Ws:
        tmp = np.zeros(
            (round_up_mod(W.shape[0], align), round_up_mod(W.shape[1], align))
        )
        tmp[: W.shape[0], : W.shape[1]] = W
        params[0, pos : pos + tmp.size] = tmp.flatten()
        pos += tmp.size
    return params.astype(np.float32)


def numpy_params_to_cuda(Ws):
    n_input_dims = Ws[0].shape[0]
    n_output_dims = Ws[-1].shape[1]
    n_neurons = Ws[-1].shape[0]
    align = 16

    size = 0
    for W in Ws:
        size += round_up_mod(W.shape[0], align) * round_up_mod(W.shape[1], align)

    params = np.zeros(shape=(size,))
    pos = 0
    for W in Ws:
        tmp = np.zeros(
            (round_up_mod(W.shape[0], align), round_up_mod(W.shape[1], align))
        )
        tmp[: W.shape[0], : W.shape[1]] = W
        params[pos : pos + tmp.size] = tmp.T.flatten()
        pos += tmp.size
    return params.astype(np.float32)


def np_net(x, Ws):
    y = x
    for W in Ws:
        y = y @ W
    return y


@pytest.mark.parametrize("n_neurons", [16, 32, 64, 128])
def test_forward_network(n_neurons):
    torch.manual_seed(42)
    # avoid padding
    n_hidden_layers = 1
    n_input_dims = 3
    n_output_dims = 5
    config = {
        "otype": "FullyFusedMLP",
        "activation": "none",
        "output_activation": "none",
        "n_neurons": n_neurons,
        "n_hidden_layers": n_hidden_layers,
        "device": device,
    }

    network = tcnn.Network(n_input_dims, n_output_dims, config)

    rng = np.random.default_rng(42)

    # Testing unpadded (input width is not padded by 1)
    Ws_size = (
        (round_up_mod(n_input_dims) * n_neurons)
        + (n_hidden_layers - 1) * n_neurons * n_neurons
        + n_neurons * n_output_dims
    )

    params = rng.uniform(-0.001, 0.001, size=Ws_size)
    x = rng.uniform(size=(1, n_input_dims))
    # params = np.ones(Ws_size) * 0.1
    # x = np.ones((1, n_input_dims))

    Ws, net_fn = cuda_params_to_numpy(network, params)

    y = net_fn(x)

    if device == "xpu":
        network.set_params(
            torch.from_numpy(numpy_params_to_xpu(Ws, n_neurons)).to(device)
        )
    else:
        network.params.data[:] = torch.from_numpy(params).to(device)

    y2 = network(torch.from_numpy(x.astype(np.float32)).to(device))
    # This passes with tiny-cuda because it pads the input tensor with ones for the first matmul
    np.testing.assert_allclose(to_numpy(y2), y, rtol=0.01, atol=1e-7)


@pytest.mark.parametrize("n_neurons", [16, 32, 64, 128])
@pytest.mark.parametrize("n_hidden_layers", [1, 2, 4, 8])
def test_padding(n_neurons, n_hidden_layers):
    n_input_dims = 3
    n_output_dims = 5
    config = {
        "otype": "FullyFusedMLP",
        "activation": "none",
        "output_activation": "none",
        "n_neurons": n_neurons,
        "n_hidden_layers": n_hidden_layers,
        "device": device,
    }

    network = tcnn.Network(n_input_dims, n_output_dims, config)

    network_params = get_reshaped_params(
        network.get_params(),
        n_neurons,
        n_neurons,
        n_neurons,
        n_hidden_layers,
        torch.float16,
        "xpu",
        "unpack",
    )
    for idx, param in enumerate(network_params):
        if idx == n_hidden_layers:
            assert np.array_equal(
                param[n_output_dims:, :].cpu(),
                np.zeros((n_neurons - n_output_dims, n_neurons)),
            )
        elif idx == 0:
            padded_ones_input = round_up_mod(n_input_dims)
            assert np.array_equal(
                param[:, padded_ones_input:].cpu(),
                np.zeros((n_neurons, n_neurons - padded_ones_input)),
            )


if __name__ == "__main__":
    # test_padding(16,1)
    test_forward_network(16)
