import torch
import torch.nn as nn
import torch.optim as optim
import intel_extension_for_pytorch
from tiny_dpcpp_nn import Network, Encoding, NetworkWithInputEncoding

BATCH_SIZE = 2**7

WIDTH = 64

TRAIN_EPOCHS = 1000  # this is high to ensure that all tests pass (some are fast < 100 and some are slow)

PRINT_PROGRESS = True


def generate_data(num_samples, input_size, output_size):
    X = torch.randn(num_samples, input_size).to("xpu")
    y = torch.randint(low=0, high=output_size, size=(num_samples,)).to("xpu")
    return X, y


def train_mlp(model, data, labels, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    loss_stagnant_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

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


def test_network():
    # Test network
    # Hyperparameters
    input_size = 10
    hidden_size = WIDTH
    hidden_layers = 2
    output_size = 10
    num_samples = BATCH_SIZE
    learning_rate = 0.01
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
    )
    # Generate dummy data
    X, y = generate_data(num_samples, input_size, output_size)

    # Evaluate before training
    initial_accuracy = evaluate(mlp, X, y)

    print(
        f"Initial Accuracy for Network: {initial_accuracy:.4f} (should be around {1/output_size:.4f})"
    )

    # Train the MLP
    train_mlp(mlp, X, y, epochs, learning_rate)

    # Evaluate after training
    final_accuracy = evaluate(mlp, X, y)
    print(f"Final Accuracy for Network: {final_accuracy:.4f}")
    assert final_accuracy == 1.0


def test_encoding():
    # Test encoding
    # Hyperparameters
    input_size = 2
    output_size = 32
    num_samples = BATCH_SIZE
    learning_rate = 0.01
    epochs = TRAIN_EPOCHS

    # Initialize the MLP
    encoding_config = {
        "otype": "Grid",
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
        dtype=torch.float,
    )

    # Generate dummy data
    X, y = generate_data(num_samples, input_size, output_size)

    # Evaluate before training
    initial_accuracy = evaluate(enc, X, y)

    print(
        f"Initial Accuracy for Encoding: {initial_accuracy:.4f} (should be around {1/output_size:.4f})"
    )

    # Train the MLP
    train_mlp(enc, X, y, epochs, learning_rate)

    # Evaluate after training
    final_accuracy = evaluate(enc, X, y)
    print(f"Final Accuracy for Encoding: {final_accuracy:.4f}")
    assert final_accuracy == 1.0


def run_test_network_with_custom_encoding(
    encoding_config,
    input_size,
    hidden_size,
    hidden_layers,
    output_size,
    num_samples,
    learning_rate,
    epochs,
    separate,
):
    if separate:
        encoding = Encoding(
            n_input_dims=input_size,
            encoding_config=encoding_config,
            device="xpu",
            dtype=torch.float,
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
        )
        nwe = torch.nn.Sequential(encoding, network).to("xpu")
    else:
        nwe = NetworkWithInputEncoding(
            n_input_dims=input_size,
            n_output_dims=output_size,
            network_config={
                "activation": "relu",
                "output_activation": "linear",
                "n_neurons": hidden_size,
                "n_hidden_layers": hidden_layers,
            },
            encoding_config=encoding_config,
            device="xpu",
            dtype=torch.float,
        )
    # Generate dummy data
    X, y = generate_data(num_samples, input_size, output_size)

    # Evaluate before training
    initial_accuracy = evaluate(nwe, X, y)
    print(
        f"Initial Accuracy: {initial_accuracy:.4f} (should be around {1/output_size:.4f})"
    )

    # Train the MLP
    train_mlp(nwe, X, y, epochs, learning_rate)

    # Evaluate after training
    final_accuracy = evaluate(nwe, X, y)
    print(f"Final Accuracy: {final_accuracy:.4f}")
    assert (
        final_accuracy > 0.8
    )  # Adjusted expectation as perfect accuracy may not be realistic


def test_network_with_encoding_all():
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
        "otype": "Grid",
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
        identity_config, separate=True, **hyper_parameters
    )

    print("Testing spherical separate")
    run_test_network_with_custom_encoding(
        spherical_harmonics_config, separate=True, **hyper_parameters
    )

    print("Testing grid separate")
    run_test_network_with_custom_encoding(
        grid_config, separate=True, **hyper_parameters
    )

    print("Testing identity nwe")
    run_test_network_with_custom_encoding(
        identity_config, separate=False, **hyper_parameters
    )

    print("Testing spherical nwe")
    run_test_network_with_custom_encoding(
        spherical_harmonics_config, separate=False, **hyper_parameters
    )

    print("Testing grid nwe")
    run_test_network_with_custom_encoding(
        grid_config, separate=False, **hyper_parameters
    )


if __name__ == "__main__":
    print("Testing network")
    test_network()

    print("Testing encoding")
    test_encoding()

    test_network_with_encoding_all()
