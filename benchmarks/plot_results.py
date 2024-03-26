import json
import matplotlib.pyplot as plt
import os
import math

FILE_PATH = os.path.join(os.path.dirname(__file__), "results")
DEVICE = "xpu"


def load_data(filename, width):
    with open(f"{FILE_PATH}/{filename}_width{width}.json", "r") as file:
        data = json.load(file)

    batch_sizes = data["batch_sizes"]
    training_throughputs = data["training_throughputs"]
    inference_throughputs = data["inference_throughputs"]

    return batch_sizes, training_throughputs, inference_throughputs


def plot_result(width):
    # Load data for both PyTorch and DPC++
    (
        batch_sizes_pow_pytorch,
        training_throughput_pytorch,
        inference_throughput_pytorch,
    ) = load_data(f"bench_result_pytorch_{DEVICE}", width)
    batch_sizes_pow_dpcpp, training_throughput_dpcpp, inference_throughput_dpcpp = (
        load_data("bench_result_dpcpp", width)
    )

    # Plotting the training throughput
    plt.figure(figsize=(8, 6))
    plt.plot(
        batch_sizes_pow_pytorch,
        training_throughput_pytorch,
        marker="o",
        label="PyTorch",
    )
    plt.plot(
        batch_sizes_pow_dpcpp, training_throughput_dpcpp, marker="x", label="DPC++"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Training Throughput (FLOPS per sample)")
    plt.title(f"Training Throughput Comparison (Width={width})")
    plt.xticks(
        batch_sizes_pow_pytorch, [f"2^{int(n)}" for n in batch_sizes_pow_pytorch]
    )
    plt.yscale("log")
    plt.legend()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "results",
            f"training_throughput_comparison_width{width}.png",
        )
    )

    # Plotting the inference throughput
    plt.figure(figsize=(8, 6))
    plt.plot(
        batch_sizes_pow_pytorch,
        inference_throughput_pytorch,
        marker="o",
        label="PyTorch",
    )
    plt.plot(
        batch_sizes_pow_dpcpp, inference_throughput_dpcpp, marker="x", label="DPC++"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Inference Throughput (FLOPS per sample)")
    plt.title(f"Inference Throughput Comparison (Width={width})")
    plt.xticks(
        batch_sizes_pow_pytorch, [f"2^{int(n)}" for n in batch_sizes_pow_pytorch]
    )
    plt.yscale("log")
    plt.legend()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "results",
            f"inference_throughput_comparison_width{width}.png",
        )
    )


if __name__ == "__main__":
    width = 64
    plot_result(width)
