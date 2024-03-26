import torch
import torch.nn.functional as F

BIAS = False


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation_func="relu",
        output_activation=None,
    ):
        super().__init__()
        self.input_width = input_size
        self.output_width = output_size
        self.layers = torch.nn.ModuleList()
        assert isinstance(activation_func, str) or None
        self.activation_func = activation_func
        self.output_activation = output_activation
        self.network_width = hidden_sizes[0]

        # Input layer
        self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0], bias=BIAS))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=BIAS)
            )

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], output_size, bias=BIAS))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = self._apply_activation(layer(x), self.output_activation)
            else:
                x = self._apply_activation(layer(x), self.activation_func)
        return x

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

    def set_weights(self):
        for layer in self.layers:
            if hasattr(layer, "weight"):
                # Standard deviation for the normal distribution
                std = 0.01
                torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
