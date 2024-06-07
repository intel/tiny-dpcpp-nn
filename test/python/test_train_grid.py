import torch
import intel_extension_for_pytorch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tiny_dpcpp_nn import Network, NetworkWithInputEncoding
from src.utils import create_models

torch.set_printoptions(precision=10)

# USE_ADAM = False
USE_ADAM = True


class SimpleSGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SimpleSGDOptimizer, self).__init__(params, defaults)

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
                p.data = p.data - group["lr"] * grad
        return loss


# Define a simple linear function for the dataset
def true_function(x):
    return 0.5 * x


# Create a synthetic dataset based on the true function
input_size = 3
output_size = 1
num_samples = 8
batch_size = 8

# Random inputs
inputs_single = torch.linspace(-1, 1, steps=num_samples)
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
device = "xpu"

model = NetworkWithInputEncoding(
    n_input_dims=input_size,
    n_output_dims=output_size,
    encoding_config={
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 14,
        "base_resolution": 16,
        "per_level_scale": 1.6,
    },
    network_config={
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 32,
        "n_hidden_layers": 2,
    },
).to(device)
# Define a loss function and an optimizer
criterion = torch.nn.MSELoss()
if USE_ADAM:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
else:
    optimizer = SimpleSGDOptimizer(model.parameters(), lr=1e-3)

# Lists for tracking loss and epochs
epoch_losses = []
epoch_count = []

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    running_loss = 0.0
    print(f"Epoch: {epoch}")
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.clone().to(device), labels.clone().to(device).to(
            torch.bfloat16
        )
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss dpcpp: {epoch_loss}")
    epoch_losses.append(epoch_loss)
    epoch_count.append(epoch + 1)

print("Finished Training")
