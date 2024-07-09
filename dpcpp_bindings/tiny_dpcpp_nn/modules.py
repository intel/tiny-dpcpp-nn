import torch
import intel_extension_for_pytorch
import numpy as np
import time

class Timer:    
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f'{self.name} {self.interval}', flush=True)

from tiny_dpcpp_nn_pybind_module import (
    Activation,
    create_network,
    create_encoding,
    create_networkwithencoding,
)

MIN_BATCH_SIZE = 8  # in tiny-dpcpp-nn the smallest possible batch size is 8


def unpad_tensor_to_input_dim(padded_tensor, output_dim):
    batch_size, current_width = padded_tensor.shape
    if output_dim > current_width:
        raise ValueError(
            "input_dim must be less than or equal to the current width of the tensor"
        )

    unpadded_tensor = padded_tensor[:, :output_dim]
    return unpadded_tensor


def get_dpcpp_activation(name):

    if name.lower() == "relu":
        activation = Activation.ReLU
    elif name.lower() == "tanh":
        activation = Activation.Tanh
    elif name.lower() == "sigmoid":
        activation = Activation.Sigmoid
    elif name.lower() == "linear" or name.lower() == "none":
        activation = Activation.Linear
    else:
        raise NotImplementedError(f"Activation: {name} not defined")

    return activation


def to_packed_layout_coord(idx, rows, cols):
    i = idx // cols
    j = idx % cols
    if (i % 2) == 0:
        return i * cols + 2 * j
    else:
        return (i - 1) * cols + 2 * j + 1


def from_packed_layout_coord(idx, rows, cols):
    # Not really used.
    i = idx // (cols * 2)
    j = idx % (cols * 2)
    if (j % 2) == 0:
        return (i * 2) * cols + j // 2
    else:
        return (i * 2 + 1) * cols + (j - 1) // 2


class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_tnn_module, input, params, info, loss_scale):
        batch_size = input.shape[0]

        if info["is_in_eval_mode"]:
            output = native_tnn_module.inference(input)
        else:
            with Timer(f'forward {input.shape}'):
                output = native_tnn_module.fwd(input)

        if batch_size > 0:
            output = output.reshape(batch_size, -1).to(input.device)
        else:  # keep shape if we have an empty input tensor with batch_size==0
            output = output.to(input.device)

        ctx.save_for_backward(input, output, params)
        ctx.info = info
        ctx.native_tnn_module = native_tnn_module
        ctx.loss_scale = loss_scale

        if "output_dim" in info:
            return output[
                ..., : info["output_dim"]
            ]  # we pad after output_dim with zeros
        else:
            # this is for encoding
            return output

    @staticmethod
    def backward(ctx, doutput):
        input, _, params = ctx.saved_tensors
        loss_scale = ctx.loss_scale

        if "width" in ctx.info:
            doutput = torch.hstack(
                (
                    doutput,
                    torch.zeros(doutput.shape[0], ctx.info["width"] - doutput.shape[1])
                    .to("xpu")
                    .to(doutput.dtype),
                )
            )

        batch_size = input.shape[0]
        doutput = doutput * loss_scale
        with Timer(f'backward {input.shape}'):
            with torch.no_grad():
                if "encoding_config" in ctx.info:
                    input_grad = None

                    if batch_size == 0:
                        grad = torch.zeros_like(params)
                    elif "width" in ctx.info:
                        # this is NWE with grid encoding
                        pack_weights = True
                        _, grad = ctx.native_tnn_module.bwd_with_encoding_grad(
                            doutput,
                            input,
                            pack_weights,
                            False,  # pack the grad, don't get dldinput
                        )
                    else:
                        pack_weights = False
                        # this is pure encoding module
                        _, grad = ctx.native_tnn_module.bwd_with_encoding_grad(
                            doutput,
                            input,
                            pack_weights,
                            False,  # don't pack the grad, don't get dldinput
                        )
                else:
                    # This is Network only
                    pack_weights = True
                    input_grad, grad = ctx.native_tnn_module.bwd_no_encoding_grad(
                        doutput, pack_weights, True  # pack the grad, get dldinput
                    )
                    if input_grad is not None:
                        input_grad = input_grad.reshape(batch_size, -1)

        grad = None if grad is None else (grad / loss_scale)
        input_grad = None if input_grad is None else (input_grad / loss_scale)

        # 4 inputs to forward, so need 4 grads
        return (None, input_grad, grad, None, None)


class Module(torch.nn.Module):
    def __init__(
        self,
        device="xpu",
        input_dtype=torch.float16,
        backend_param_dtype=torch.float16,
    ):
        super(Module, self).__init__()
        self.device = device
        self.input_dtype = input_dtype
        self.backend_param_dtype = backend_param_dtype

        if backend_param_dtype == torch.float16:
            self.loss_scale = 128.0
        else:
            self.loss_scale = 1.0

        self.tnn_module = self.create_module()

        if self.tnn_module.n_params():
            initial_params = self.tnn_module.initial_params()
            # This explicitely creates a tensor whose memory is managed by PyTorch in modules.py
            cloned_params = initial_params.clone().detach().to(torch.float32)
            self.params = torch.nn.Parameter(cloned_params, requires_grad=True)
        else:
            print(
                "No params initialised, as n_params = 0. This is correct for Encodings (apart from grid encodings)."
            )
            self.params = torch.nn.Parameter(torch.zeros(1), requires_grad=False).to(
                self.device
            )

    def set_params(self, params=None):
        if not self.tnn_module.n_params():
            return

        packed = params is None

        if params is None:
            # this forces the backend to use the self.params which were overwritten in python only (pointing to different backend arrays)
            params = self.params

        assert isinstance(params, torch.Tensor), "Params is not a torch.Tensor"

        self.tnn_module.set_params(params.to(self.backend_param_dtype), packed)

        if not packed:
            packed_weights = self.get_params()  # this is always packed params
            # Set self.params to the params passed. Backend dpcpp and python seem to be different underlying memories
            self.params.data.copy_(packed_weights)

    def get_params(self):
        # return packed params. Currently tnn_module.initial_params() does the same as get_params()
        return self.tnn_module.get_params()

    def get_reshaped_params(
        self,
        weights=None,
        is_packed_format=True,
        is_transposed=True,
    ):
        if not is_transposed:
            raise RuntimeError(
                "All matrices should be transposed by now, please check first"
            )
        all_weights = []
        if weights is None:
            weights = self.params

        n_input_dims = (
            self.width if self.n_input_dims <= self.width else self.n_input_dims
        )  # because we pad
        input_matrix = (
            torch.zeros(self.width, n_input_dims)
            .to(self.backend_param_dtype)
            .to(self.device)
        )

        for i in range(n_input_dims):
            for j in range(self.width):
                if is_packed_format:
                    idx = to_packed_layout_coord(
                        i * self.width + j, n_input_dims, self.width
                    )
                else:
                    idx = i * self.width + j
                if is_transposed:
                    input_matrix[j, i] = weights[idx]
                else:
                    input_matrix[i, j] = weights[idx]

        len_input_matrix = input_matrix.shape[0] * input_matrix.shape[1]
        hidden_layer_size = self.width * self.width
        hidden_matrices = []

        for nth_hidden in range(self.n_hidden_layers - 1):
            hidden_matrix = (
                torch.zeros(self.width, self.width)
                .to(self.backend_param_dtype)
                .to(self.device)
            )

            for i in range(self.width):
                for j in range(self.width):
                    if is_packed_format:
                        idx = to_packed_layout_coord(
                            i * self.width + j, self.width, self.width
                        )
                    else:
                        idx = i * self.width + j
                    if is_transposed:
                        hidden_matrix[j, i] = weights[
                            len_input_matrix + nth_hidden * hidden_layer_size + idx
                        ]
                    else:
                        hidden_matrix[i, j] = weights[
                            len_input_matrix + nth_hidden * hidden_layer_size + idx
                        ]
            hidden_matrices.append(hidden_matrix)

        output_matrix = (
            torch.zeros(self.width, self.width)
            .to(self.backend_param_dtype)
            .to(self.device)
        )

        for i in range(self.width):
            for j in range(
                self.width
            ):  # the weights in Swiftnet are padded to width with zeros
                if is_packed_format:
                    idx = to_packed_layout_coord(
                        i * self.width + j, self.width, self.width
                    )
                else:
                    idx = i * self.width + j
                if is_transposed:
                    output_matrix[j, i] = weights[
                        len_input_matrix
                        + (self.n_hidden_layers - 1) * hidden_layer_size
                        + idx
                    ]
                else:
                    output_matrix[i, j] = weights[
                        len_input_matrix
                        + (self.n_hidden_layers - 1) * hidden_layer_size
                        + idx
                    ]

        all_weights.append(input_matrix)
        all_weights.extend(hidden_matrices)
        all_weights.append(output_matrix[: self.n_output_dims, ...])

        return all_weights

    def forward(self, x):
        batch_size, input_dim = x.shape
        padded_batch_size = (
            (batch_size + MIN_BATCH_SIZE - 1) // MIN_BATCH_SIZE * MIN_BATCH_SIZE
        )

        padded_tensor = (
            x
            if batch_size == padded_batch_size
            else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
        ).to(dtype=self.input_dtype)

        if self.name == "network":
            padded_tensor = (
                padded_tensor
                if input_dim == self.width
                else torch.nn.functional.pad(
                    padded_tensor, [0, self.width - input_dim, 0, 0]
                )
            ).to(dtype=self.input_dtype)
        info = {"is_in_eval_mode": not self.training}

        if hasattr(self, "n_hidden_layers"):
            # added for NWE and Network
            info.update(
                {
                    "n_hidden_layers": self.n_hidden_layers - 1,
                    "input_dim": self.n_input_dims,
                    "output_dim": self.n_output_dims,
                    "width": self.width,
                }
            )

        if hasattr(self, "encoding_config"):
            # added for NWE and encoding
            info.update({"encoding_config": self.encoding_config})

        self.set_params()
        output = _module_function.apply(
            self.tnn_module,
            padded_tensor.contiguous(),
            self.params,
            info,
            self.loss_scale,
        )
        return output[:batch_size, ...]


class Network(Module):
    def __init__(
        self,
        n_input_dims,
        n_output_dims,
        network_config,
        device="xpu",
        input_dtype=torch.float16,
        backend_param_dtype=torch.float16,
    ):
        self.network_config = network_config

        self.width = self.network_config["n_neurons"]
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_hidden_layers = self.network_config["n_hidden_layers"]
        self.activation = get_dpcpp_activation(self.network_config["activation"])
        self.output_activation = get_dpcpp_activation(
            self.network_config["output_activation"]
        )
        self.name = "network"
        super().__init__(
            device=device,
            input_dtype=input_dtype,
            backend_param_dtype=backend_param_dtype,
        )

    def create_module(self):
        return create_network(
            self.n_input_dims,
            self.n_output_dims,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.width,
            str(self.backend_param_dtype),
        )


class NetworkWithInputEncoding(Module):
    def __init__(
        self,
        n_input_dims,
        n_output_dims,
        network_config,
        encoding_config,
        device="xpu",
        input_dtype=torch.float,
        backend_param_dtype=torch.float16,
    ):
        self.network_config = network_config

        self.width = self.network_config["n_neurons"]
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_hidden_layers = self.network_config["n_hidden_layers"]
        self.activation = get_dpcpp_activation(self.network_config["activation"])
        self.output_activation = get_dpcpp_activation(
            self.network_config["output_activation"]
        )
        self.name = "network_with_encoding"

        self.encoding_config = encoding_config
        self.encoding_name = self.encoding_config["otype"]

        if "n_dims_to_encode" not in self.encoding_config:
            self.encoding_config["n_dims_to_encode"] = self.n_input_dims
        assert (
            input_dtype == torch.float
        ), f"Currently only torch.float supported as input_dtype. {input_dtype} was chosen instead"
        super().__init__(
            device=device,
            input_dtype=input_dtype,
            backend_param_dtype=backend_param_dtype,
        )

    def create_module(self):

        return create_networkwithencoding(
            self.n_input_dims,
            self.n_output_dims,
            self.n_hidden_layers,
            self.activation,
            self.output_activation,
            self.encoding_config,
            self.width,
            str(self.backend_param_dtype),
        )


class Encoding(Module):
    def __init__(
        self,
        n_input_dims,
        encoding_config,
        device="xpu",
        input_dtype=torch.float,
        backend_param_dtype=torch.float,
    ):
        self.n_input_dims = n_input_dims

        self.encoding_config = encoding_config

        self.encoding_name = self.encoding_config["otype"]
        self.name = "encoding"

        if "n_dims_to_encode" not in self.encoding_config:
            self.encoding_config["n_dims_to_encode"] = self.n_input_dims

        assert (
            input_dtype == torch.float
        ), "Only torch.float is currently supported for encoding"
        assert (
            backend_param_dtype == torch.float
        ), "Only torch.float is currently supported for encoding"
        # Spherical and identity can have non-float, but grid encoding needs
        # float, as half precision atomics are currently not supported in SYCL
        super().__init__(
            device=device,
            input_dtype=input_dtype,
            backend_param_dtype=backend_param_dtype,
        )

        self.n_output_dims = self.tnn_module.n_output_dims()

    def create_module(self):
        return create_encoding(
            self.encoding_name,
            self.encoding_config,
        )
