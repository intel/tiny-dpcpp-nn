import torch
import intel_extension_for_pytorch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.utils import create_models

torch.set_printoptions(precision=10)

CALC_DPCPP = True
CALC_REF = True

USE_ADAM = True

DTYPE = torch.bfloat16
WIDTH = 16
num_epochs = 100


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
        grad_sum = 0.0
        param_sum = 0.0
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    print("p.grad is none")
                    continue
                grad = p.grad.data
                # if grad.shape[1] == 1:
                #     print("dpcpp grad: ")
                #     grad_last_layer_reshaped = grad[-256:, 0].reshape(16, 16)
                #     print(grad_last_layer_reshaped)
                #     print("dpcpp param: ")
                #     param_last_layer_reshaped = p.data[-256:, 0].reshape(16, 16)
                #     print(param_last_layer_reshaped)

                p.data = p.data - group["lr"] * grad

                grad_sum += torch.abs(grad).sum()
                param_sum += torch.abs(p.data).sum()
        print(f"{self.name} Grad sum: {grad_sum}")
        print(f"{self.name} p.data sum: {param_sum}")
        return loss


# Define a simple linear function for the dataset
def true_function(x):
    return 0.5 * x


# Create a synthetic dataset based on the true function
input_size = WIDTH
output_size = 1
num_samples = 2**10
batch_size = 2**10

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
model_dpcpp, model_torch = create_models(
    input_size, [WIDTH], output_size, "relu", "linear", dtype=DTYPE
)

model_torch.to(DTYPE).to("xpu")


def criterion(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


if USE_ADAM:
    if CALC_DPCPP:
        optimizer1 = torch.optim.Adam(model_dpcpp.parameters(), lr=1e-2)
    if CALC_REF:
        optimizer2 = torch.optim.Adam(model_torch.parameters(), lr=1e-2)
else:
    if CALC_DPCPP:
        optimizer1 = SimpleSGDOptimizer(model_dpcpp.parameters(), name="dpcpp", lr=1e-2)
    if CALC_REF:
        optimizer2 = SimpleSGDOptimizer(model_torch.parameters(), name="torch", lr=1e-2)

# Lists for tracking loss and epochs
epoch_losses1 = []
epoch_losses2 = []
epoch_count = []

# Training loop
for epoch in range(num_epochs):
    running_loss1 = 0.0
    running_loss2 = 0.0
    print(f"Epoch: {epoch}")
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        if CALC_DPCPP:
            inputs1, labels1 = inputs.clone().to(device), labels.clone().to(device).to(
                DTYPE
            )
            # Forward pass
            outputs1 = model_dpcpp(inputs1)
            loss1 = criterion(outputs1, labels1)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            running_loss1 += loss1.item()
        if CALC_REF:
            inputs2, labels2 = inputs.clone().to(device).to(DTYPE).to(
                "xpu"
            ), labels.clone().to(device).to(DTYPE).to("xpu")
            outputs2 = model_torch(inputs2)
            loss2 = criterion(outputs2, labels2)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            running_loss2 += loss2.item()

    if CALC_DPCPP:
        epoch_loss1 = running_loss1 / len(dataloader)
        print(f"Epoch {epoch+1}, Loss dpcpp: {epoch_loss1}")
        epoch_losses1.append(epoch_loss1)
    if CALC_REF:
        epoch_loss2 = running_loss2 / len(dataloader)
        print(f"Epoch {epoch+1}, Loss torch: {epoch_loss2}")
        epoch_losses2.append(epoch_loss2)
    print("================================")
    epoch_count.append(epoch + 1)

print("Finished Training")

# Plot the loss over epochs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epoch_count, epoch_losses1, label="Training Loss Torch")
plt.plot(epoch_count, epoch_losses2, label="Training Loss DPCPP")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

# Plot the ground truth and the learned function
plt.subplot(1, 2, 2)
plt.scatter(
    inputs_training.cpu().to(torch.float)[:, 0],
    labels_training.cpu().to(torch.float)[:, 0],
    s=8,
    label="Ground Truth",
)
if CALC_DPCPP:
    with torch.no_grad():
        learned_function_dpcpp = model_dpcpp(inputs1.to(device)).cpu()
    # print(inputs1.shape)
    # print(learned_function_dpcpp.shape)
    plt.scatter(
        inputs_training.cpu().to(torch.float)[:, 0],
        learned_function_dpcpp.cpu().to(torch.float)[:, 0],
        s=8,
        label="Learned Function dpcpp",
    )
if CALC_REF:
    with torch.no_grad():
        learned_function_torch = model_torch(inputs_training.to(device)).cpu()
    plt.scatter(
        inputs_training.cpu().to(torch.float)[:, 0],
        learned_function_torch.cpu().to(torch.float)[:, 0],
        s=8,
        label="Learned Function torch",
    )
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Ground Truth vs Learned Function")
plt.legend()
plt.ylim(-1, 1)
plt.tight_layout()

# Save the figure instead of showing it
plt.savefig("loss_and_function.png")
