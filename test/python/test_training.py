import torch
import pytest
import intel_extension_for_pytorch
from torch.utils.data import DataLoader, TensorDataset
from src.utils import create_models
from tiny_dpcpp_nn import Network, Encoding

torch.set_printoptions(precision=10)

optimisers = [
    "sgd"
]  # we test only against sgd, as we can compare against the internal variables here
dtypes = [torch.bfloat16]
# dtypes = [torch.float16, torch.bfloat16]
TRAIN_EPOCHS = 10000  # this is high to ensure that all tests pass (some are fast < 100 and some are slow)

USE_NWE = False
WIDTH = 32
num_epochs = 100
DEVICE = "xpu"

BATCH_SIZE = 2**7
LR = 1e-3
PRINT_PROGRESS = True


# Self defined SGD for debugging purposes
class SimpleSGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, name="", lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SimpleSGDOptimizer, self).__init__(params, defaults)
        self.name = name

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    print("p.grad is none")
                    continue
                grad = p.grad.data

                p.data.copy_(p.data - group["lr"] * grad)

        return loss


# Define a simple linear function for the dataset
def true_function(x):
    return 0.5 * x


# Test grads over multiple iterations
@pytest.mark.parametrize(
    "dtype, optimiser",
    [(dtype, optimiser) for dtype in dtypes for optimiser in optimisers],
)
def test_regression(dtype, optimiser):
    # Create a synthetic dataset based on the true function
    input_size = WIDTH
    output_size = 1
    num_samples = 2**3
    batch_size = 2**3

    # inputs
    # inputs_single = torch.linspace(-1, 1, steps=num_samples)
    inputs_single = torch.ones(num_samples)
    inputs_training = inputs_single.repeat(input_size, 1).T

    # Corresponding labels with some noise
    noise = torch.randn(num_samples, output_size) * 0.0

    labels_training = true_function(inputs_training) + noise
    # Create a DataLoader instance for batch processing
    dataset = TensorDataset(inputs_training, labels_training)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )  # if we shuffle, the loss is not identical

    # Instantiate the network
    model_dpcpp, model_torch = create_models(
        input_size,
        [WIDTH],
        output_size,
        "relu",
        "linear",
        use_nwe=USE_NWE,
        dtype=dtype,
        use_weights_of_tinynn=True,
        store_params_as_full_precision=False,
    )

    def criterion(y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    if optimiser == "adam":
        optimizer1 = torch.optim.Adam(model_dpcpp.parameters(), lr=LR)
        optimizer2 = torch.optim.Adam(model_torch.parameters(), lr=LR)
    elif optimiser == "sgd":
        optimizer1 = SimpleSGDOptimizer(model_dpcpp.parameters(), name="dpcpp", lr=LR)
        optimizer2 = SimpleSGDOptimizer(model_torch.parameters(), name="torch", lr=LR)
    else:
        raise NotImplementedError(f"{optimiser} not implemented as optimisers")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            # DPCPP
            inputs1 = inputs.clone().to(DEVICE).to(dtype)
            labels1 = labels.clone().to(DEVICE).to(dtype)
            # Forward pass
            outputs1 = model_dpcpp(inputs1)
            loss1 = criterion(outputs1, labels1)
            optimizer1.zero_grad()
            loss1.backward()
            params1 = model_dpcpp.params.clone().detach()

            grads1 = model_dpcpp.params.grad.clone().detach()
            optimizer1.step()
            params_updated1 = model_dpcpp.params.clone().detach()
            assert not torch.equal(
                params1, params_updated1
            ), "The params for model_dpcpp are the same after update, but they should be different."

            # Torch
            inputs2 = inputs.clone().to(DEVICE).to(dtype)
            labels2 = labels.clone().to(DEVICE).to(dtype)
            outputs2 = model_torch(inputs2)
            loss2 = criterion(outputs2, labels2)
            optimizer2.zero_grad()
            loss2.backward()
            params2 = model_torch.get_all_weights()
            grads2 = model_torch.get_all_grads()
            optimizer2.step()
            params_updated2 = model_torch.get_all_weights()

            assert not torch.equal(
                params2, params_updated2
            ), "The params for model_dpcpp are the same after update, but they should be different."

            # Assertions
            assert (
                params1.dtype == params2.dtype
            ), f"Params not same dtype: {params1.dtype}, {params2.dtype}"
            assert torch.isclose(
                inputs1, inputs2
            ).all(), f"Inputs not close with sums: {abs(inputs1).sum()}, {abs(inputs2).sum()}"
            assert torch.isclose(
                outputs1, outputs2
            ).all(), f"Outputs not close with sums: {abs(outputs1).sum()}, {abs(outputs2).sum()}"
            assert torch.isclose(
                labels1, labels2
            ).all(), f"Labels not close with sums: {abs(labels1).sum()}, {abs(labels2).sum()}"
            assert torch.isclose(
                loss1, loss2
            ).all(), f"Loss not close with sums: {loss1.item()}, {loss2.item()}"
            assert torch.isclose(
                abs(params1).sum(), abs(params2).sum()
            ), f"Params before not close with sums: {abs(params1).sum()}, {abs(params2).sum()}"

            assert torch.isclose(
                abs(grads1).sum(), abs(grads2).sum()
            ), f"Grads not close with sums: {abs(grads1).sum()}, {abs(grads2).sum()}"

            assert torch.isclose(
                abs(params_updated1).sum(), abs(params_updated2).sum()
            ), f"Params after not close with sums: {abs(params_updated1).sum()}, {abs(params_updated2).sum()}"

        print(f"Epoch {epoch}, Losses (dpcpp/torch): { loss1.item()}/{ loss2.item()}")
        print(
            "========================================================================"
        )

    print("Finished Training")


# Testing classification and convergence of all Network/NetworkWithEncodings
def generate_data(num_samples, input_size, output_size):
    X = torch.randn(num_samples, input_size).to("xpu")
    y = torch.randint(low=0, high=output_size, size=(num_samples,)).to("xpu")
    return X, y


def train_mlp(model, data, labels, epochs, learning_rate, optimiser):
    criterion = torch.nn.CrossEntropyLoss()

    if optimiser == "adam":
        print("Using ADAM")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimiser == "sgd":
        print("Using SGD")
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = SimpleSGDOptimizer(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"{optimiser} not implemented.")

    best_loss = float("inf")
    loss_stagnant_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        for param in model.parameters():
            if torch.isnan(param).any():
                print("NaN value detected in parameters!")
                break
            if torch.isnan(param.grad).any():
                print("NaN value detected in grad!")
                break

        optimizer.step()  # Update weights
        for param in model.parameters():
            if torch.isnan(param).any():
                print("NaN value detected in parameters after!")
                break
            if torch.isnan(param.grad).any():
                print("NaN value detected in grad after!")
                break

        # Early stopping condition
        if loss.item() < best_loss:
            best_loss = loss.item()
            loss_stagnant_counter = 0
        else:
            loss_stagnant_counter += 1

        if loss_stagnant_counter >= 50:
            print("Loss hasn't improved for 50 epochs. Training aborted.")
            break

        if PRINT_PROGRESS and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def evaluate(model, data, labels):
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
    return accuracy


@pytest.mark.parametrize(
    "dtype, optimiser",
    [(dtype, optimiser) for dtype in dtypes for optimiser in optimisers],
)
def test_network(dtype, optimiser):
    # Test network
    # Hyperparameters
    input_size = WIDTH
    hidden_size = WIDTH
    hidden_layers = 2
    output_size = WIDTH
    num_samples = BATCH_SIZE
    learning_rate = 0.1
    epochs = TRAIN_EPOCHS

    # Initialize the MLP
    network_config = {
        "activation": "relu",
        "output_activation": "linear",
        "n_neurons": hidden_size,
        "n_hidden_layers": hidden_layers,
    }
    mlp = Network(
        n_input_dims=input_size,
        n_output_dims=output_size,
        network_config=network_config,
        dtype=dtype,
    )
    # Generate dummy data
    X, y = generate_data(num_samples, input_size, output_size)

    # Evaluate before training
    initial_accuracy = evaluate(mlp, X, y)

    print(
        f"Initial Accuracy for Network: {initial_accuracy:.4f} (should be around {1/output_size:.4f})"
    )

    # Train the MLP
    train_mlp(mlp, X, y, epochs, learning_rate, optimiser)

    # Evaluate after training
    final_accuracy = evaluate(mlp, X, y)
    print(f"Final Accuracy for Network: {final_accuracy:.4f}")
    assert final_accuracy >= 0.9


@pytest.mark.parametrize(
    "dtype, optimiser",
    [(dtype, optimiser) for dtype in dtypes for optimiser in optimisers],
)
def test_encoding(dtype, optimiser):
    # Test encoding
    # Hyperparameters
    input_size = 2
    output_size = 32
    num_samples = BATCH_SIZE
    learning_rate = 0.01
    epochs = TRAIN_EPOCHS

    # Initialize the MLP
    encoding_config = {
        "otype": "HashGrid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 15,
        "base_resolution": 16,
        "per_level_scale": 1.5,
    }

    enc = Encoding(
        n_input_dims=input_size,
        encoding_config=encoding_config,
        device="xpu",
    )

    # Generate dummy data
    X, y = generate_data(num_samples, input_size, output_size)

    # Evaluate before training
    initial_accuracy = evaluate(enc, X, y)

    print(
        f"Initial Accuracy for Encoding: {initial_accuracy:.4f} (should be around {1/output_size:.4f})"
    )

    # Train the MLP
    train_mlp(enc, X, y, epochs, learning_rate, optimiser)

    # Evaluate after training
    final_accuracy = evaluate(enc, X, y)
    print(f"Final Accuracy for Encoding: {final_accuracy:.4f}")
    assert final_accuracy > 0.8


def run_test_network_with_custom_encoding(
    encoding_config,
    dtype,
    input_size,
    hidden_size,
    hidden_layers,
    output_size,
    num_samples,
    learning_rate,
    epochs,
    optimiser,
):
    encoding = Encoding(
        n_input_dims=input_size,
        encoding_config=encoding_config,
        device="xpu",
    )
    network = Network(
        n_input_dims=encoding.n_output_dims,
        n_output_dims=output_size,
        network_config={
            "activation": "relu",
            "output_activation": "linear",
            "n_neurons": hidden_size,
            "n_hidden_layers": hidden_layers,
        },
        dtype=dtype,
    )
    nwe = torch.nn.Sequential(encoding, network).to("xpu")
    # Generate dummy data
    X, y = generate_data(num_samples, input_size, output_size)

    # Evaluate before training
    initial_accuracy = evaluate(nwe, X, y)
    print(
        f"Initial Accuracy: {initial_accuracy:.4f} (should be around {1/output_size:.4f})"
    )

    # Train the MLP
    train_mlp(nwe, X, y, epochs, learning_rate, optimiser)

    # Evaluate after training
    final_accuracy = evaluate(nwe, X, y)
    print(f"Final Accuracy: {final_accuracy:.4f}")
    assert final_accuracy > 0.8


@pytest.mark.parametrize(
    "dtype",
    [(dtype) for dtype in dtypes],
)
def test_network_with_encoding_all(dtype):
    optimiser = "adam"
    input_size = 3
    spherical_harmonics_config = {
        "otype": "SphericalHarmonics",
        "degree": 4,
    }

    identity_config = {
        "otype": "Identity",
        "n_dims_to_encode": input_size,  # assuming the input size is 2 as in other tests
        "scale": 1.0,
        "offset": 0.0,
    }

    grid_config = {
        "otype": "HashGrid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.5,
        "interpolation": "Linear",
    }

    hyper_parameters = {
        "input_size": input_size,
        "hidden_size": WIDTH,
        "hidden_layers": 2,
        "output_size": int(WIDTH / 2),
        "num_samples": BATCH_SIZE,
        "learning_rate": 0.01,
        "epochs": TRAIN_EPOCHS,
    }

    print("Testing identity separate")
    run_test_network_with_custom_encoding(
        identity_config, dtype, optimiser=optimiser, **hyper_parameters
    )

    print("Testing spherical separate")
    run_test_network_with_custom_encoding(
        spherical_harmonics_config,
        dtype,
        optimiser=optimiser,
        **hyper_parameters,
    )

    print("Testing grid separate")
    run_test_network_with_custom_encoding(
        grid_config, dtype, optimiser=optimiser, **hyper_parameters
    )


if __name__ == "__main__":
    dtype = torch.float16
    optimiser = "sgd"
    test_regression(dtype, optimiser)

    optimiser = "sgd"
    dtype = torch.float16
    print("Testing network")
    test_network(dtype, optimiser)

    print("Testing encoding")
    test_encoding()

    print("test_network_with_encoding_all")
    test_network_with_encoding_all(dtype)
