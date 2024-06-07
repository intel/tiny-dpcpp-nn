import numpy as np
import torch

import intel_extension_for_pytorch
import pytest
import pdb
from src.utils import create_models, compare_matrices, get_grad_params

input_sizes = [8]
output_funcs = ["linear", "relu", "sigmoid"]
output_sizes = [8]
activation_funcs = ["relu", "linear", "sigmoid"]
hidden_layer_counts = [1, 2, 4]
# hidden_sizes = [16, 32, 64, 128]
hidden_sizes = [64]

BATCH_SIZE = 2**10
DEVICE_NAME = "xpu"


class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predicted, target):
        # Calculate the mean squared error
        mse = torch.mean((predicted - target) ** 2)
        return mse


def train_model(model, x_train, y_train, n_steps):
    batch_size = BATCH_SIZE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = CustomMSELoss()
    # loss_fn = torch.nn.MSELoss()
    y_predicted_all = []
    grads = []
    params = []
    losses = []
    for n in range(n_steps):
        all_loss = []
        for idx in range(x_train.shape[0] // batch_size):
            y_pred = model(x_train[idx * batch_size : (idx + 1) * batch_size, ...])
            loss = loss_fn(
                y_pred,
                y_train[idx * batch_size : (idx + 1) * batch_size],
            ).to(DEVICE_NAME)
            y_predicted_all.append(y_pred.detach().cpu().to(torch.float))
            optimizer.zero_grad()
            loss.backward()

            grads_all, params_all = get_grad_params(model)
            grads.append(grads_all)
            params.append(params_all)

            optimizer.step()
            all_loss.append(loss.detach().cpu().to(torch.float))
        loss_mean = np.mean(np.array(all_loss))
        losses.append(loss_mean)
        # print(f"{n} - Loss: {loss_mean}")

    return losses, y_predicted_all, grads, params


@pytest.mark.parametrize(
    "input_size, hidden_size, hidden_layers, output_size, activation_func, output_func",
    [
        (
            input_size,
            hidden_size,
            hidden_layers,
            output_size,
            activation_func,
            output_func,
        )
        for input_size in input_sizes
        for hidden_layers in hidden_layer_counts
        for hidden_size in hidden_sizes
        for output_size in output_sizes
        for activation_func in activation_funcs
        for output_func in output_funcs
    ],
)
def test_grad(
    input_size,
    hidden_size,
    hidden_layers,
    output_size,
    activation_func,
    output_func,
    iterations=2,
    n_steps=1,  # if this is too large, there will be accumulated error (weights aren't the same, thus the loss is not the same etc)
):
    for iter_ in range(iterations):
        print(f"Starting iteration {iter_}")
        if iter_ == 0:
            # easiest, debug test
            x_train = (
                torch.tensor(BATCH_SIZE * [0.001 for _ in range(input_size)])
                .to(DEVICE_NAME)
                .reshape(BATCH_SIZE, -1)
            )
            y_train = torch.ones([BATCH_SIZE, output_size]).to(DEVICE_NAME)
        else:
            x_train = torch.rand([BATCH_SIZE, input_size]).to(DEVICE_NAME)
            y_train = torch.rand([BATCH_SIZE, output_size]).to(DEVICE_NAME)

        # Need to generate new model, because weights are updated in one loop.
        model_dpcpp, model_torch = create_models(
            input_size,
            [hidden_size] * hidden_layers,
            output_size,
            activation_func,
            output_func,
        )

        loss_dpcpp, y_dpcpp, grads_dpcpp, params_dpcpp = train_model(
            model_dpcpp, x_train, y_train, n_steps
        )
        loss_torch, y_torch, grads_torch, params_torch = train_model(
            model_torch, x_train, y_train, n_steps
        )

        params_dpcpp = params_dpcpp[0][0]
        params_torch = params_torch[0]
        compare_matrices(params_dpcpp, params_torch)

        grads_dpcpp = grads_dpcpp[0][0]
        grads_torch = grads_torch[0]
        assert len(grads_dpcpp) == len(grads_torch)
        for layer in range(len(grads_dpcpp)):
            assert (
                torch.abs(grads_dpcpp[layer]).sum()
                - torch.abs(grads_dpcpp[layer]).sum()
            ) < 1e-3
        compare_matrices(grads_dpcpp, grads_torch)


@pytest.mark.parametrize(
    "input_size, hidden_size, hidden_layers, output_size, activation_func, output_func",
    [
        (
            input_size,
            hidden_size,
            hidden_layers,
            output_size,
            activation_func,
            output_func,
        )
        for input_size in input_sizes
        for hidden_layers in hidden_layer_counts
        for hidden_size in hidden_sizes
        for output_size in output_sizes
        for activation_func in activation_funcs
        for output_func in output_funcs
    ],
)
def test_fwd(
    input_size, hidden_size, hidden_layers, output_size, activation_func, output_func
):
    # Generate random input data for testing
    torch.manual_seed(123)
    input_data = torch.randn(BATCH_SIZE, input_size).to(DEVICE_NAME)
    model_dpcpp, model_torch = create_models(
        input_size,
        [hidden_size] * hidden_layers,
        output_size,
        activation_func,
        output_func,
    )
    model_torch.to(DEVICE_NAME)
    model_dpcpp.to(DEVICE_NAME)
    # print("Params model_torch: ", list(model_torch.parameters())[:10])
    # print("Params model_dpcpp: ", list(model_dpcpp.parameters())[:10])
    y_torch = model_torch(input_data)
    y_dpcpp = model_dpcpp(input_data)
    print("Torch output: ", y_torch[-1, :])
    print("DPCPP output: ", y_dpcpp[-1, :])

    if abs(y_torch.sum() - y_dpcpp.sum()) / (abs(y_torch).sum()) > 0.01:
        print(
            f"diff: {y_torch[-1, :] - y_dpcpp[-1, :]}, average: {abs(y_torch - y_dpcpp).mean()}"
        )
    assert (
        abs(y_torch.sum() - y_dpcpp.sum()) / (abs(y_torch).sum()) < 0.01
    ), f"Forward error is too large {abs(y_torch.sum() - y_dpcpp.sum()) / (abs(y_torch).sum()) :.4f}"


if __name__ == "__main__":
    input_width = 8
    output_width = 16
    hidden_size = 16
    hidden_layers = 2
    # activation_func = "linear"
    activation_func = "relu"
    output_func = "linear"
    # output_func = "sigmoid"

    test_fwd(
        input_width,
        hidden_size,
        hidden_layers,
        output_width,
        activation_func,
        output_func,
    )
    print("Passed fwd test")

    test_grad(
        input_width,
        hidden_size,
        hidden_layers,
        output_width,
        activation_func,
        output_func,
    )
    print("Passed bwd test")
