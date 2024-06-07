import torch
import torch.nn.functional as F
import copy
import numpy as np

BIAS = False


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation_func="relu",
        output_activation=None,
        use_batchnorm=False,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.dtype = dtype
        # Used for gradecheck and naming consistency with modules.py (Swiftnet)
        self.input_width = input_size
        self.output_width = output_size
        # if input_size < 16:
        #     print("Currently we do manual encoding for input size < 16.")
        #     hidden_sizes.insert(0, 64)
        self.layers = torch.nn.ModuleList()
        assert isinstance(activation_func, str) or None
        self.activation_func = activation_func
        self.output_activation = output_activation
        self.use_batchnorm = use_batchnorm

        # Input layer
        input_dim = hidden_sizes[0] if input_size <= 64 else input_size
        self.layers.append(
            torch.nn.Linear(input_dim, hidden_sizes[0], bias=BIAS).to(self.dtype)
        )
        # if input_size < 16:
        #     # the encoding in the current implementaiton doesn't have grad.
        #     # Set requires_grad to False for the parameters of the first layer (layers[0])
        #     self.layers[0].weight.requires_grad = False

        if self.use_batchnorm:
            self.layers.append(torch.nn.BatchNorm1d(hidden_sizes[0]).to(self.dtype))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=False).to(
                    self.dtype
                )
            )

            # BatchNorm layer for hidden layers (if enabled)
            if self.use_batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(hidden_sizes[i]).to(self.dtype))

        # Output layer
        self.layers.append(
            torch.nn.Linear(hidden_sizes[-1], output_size, bias=False).to(self.dtype)
        )

    def forward(self, x):
        x_changed_dtype = x.to(self.dtype)
        assert x_changed_dtype.dtype == self.dtype
        batch_size = x_changed_dtype.size(0)
        ones = torch.ones(
            (batch_size, self.layers[0].in_features - self.input_width),
            dtype=x_changed_dtype.dtype,
            device=x_changed_dtype.device,
        )
        x_changed_dtype = torch.cat((x_changed_dtype, ones), dim=1)

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x_changed_dtype = self._apply_activation(
                    layer(x_changed_dtype), self.output_activation
                )
            else:
                x_changed_dtype = self._apply_activation(
                    layer(x_changed_dtype), self.activation_func
                )

        return x_changed_dtype

    def _apply_activation(self, x, activation_func):
        if activation_func == "relu":
            return F.relu(x)
        elif activation_func == "leaky_relu":
            return F.leaky_relu(x)
        elif activation_func == "sigmoid":
            return torch.sigmoid(x)
        elif activation_func == "tanh":
            return torch.tanh(x)
        elif (
            (activation_func == "None")
            or (activation_func is None)
            or (activation_func == "linear")
        ):
            return x
        else:
            raise ValueError("Invalid activation function")

    def set_weights(self, parameters):
        for i, weight in enumerate(parameters):
            assert self.layers[i].weight.shape == weight.shape
            self.layers[i].weight = torch.nn.Parameter(weight)
