from src.mlp_benchmark import MLP

import torch
import numpy as np
import time
import json

# DEVICE_NAME = "cuda"
DEVICE_NAME = "xpu"

# USE_TINY_NN = False
USE_TINY_NN = True

DTYPE = torch.bfloat16

if DEVICE_NAME == "xpu":
    import intel_extension_for_pytorch as ipex  # required for xpu support

    OPTIMISE_MODEL = False  # not supported
else:
    OPTIMISE_MODEL = False


def start_training(
    input_size,
    hidden_sizes,
    output_size,
    batch_sizes,
    path=None,
    debug=False,
):
    activation_func = "relu"
    output_func = None

    if USE_TINY_NN:
        from tiny_dpcpp_nn import Network

        network_config = {
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_sizes[0],
            "n_hidden_layers": len(hidden_sizes),
        }
        model = Network(
            n_input_dims=input_size,
            n_output_dims=output_size,
            network_config=network_config,
            input_dtype=DTYPE,
            backend_param_dtype=DTYPE,
        )
    else:
        model = MLP(
            input_size,
            hidden_sizes,
            output_size,
            activation_func,
            output_func,
        ).to(DEVICE_NAME)
        model.set_weights()
        model.to(DTYPE)

    # Run the network
    bench_result = {
        "batch_sizes": [],
        "training_throughputs": [],
        "inference_throughputs": [],
        "training_time": [],
        "inference_time": [],
    }

    for batch_size in batch_sizes:
        bench_result["batch_sizes"].append(np.log2(batch_size))
        print(f"Running batch size: {np.log2(batch_size)}")
        N_ITERS = 1000
        WARMUP_ITERS = N_ITERS / 4

        throughputs = []
        model.train()
        if OPTIMISE_MODEL:
            model_torch, optimized_optimizer = ipex.optimize(
                model, optimizer=torch.optim.Adam
            )
        else:
            model_torch = model

        timer_start = time.perf_counter()
        elapsed_times = []
        time_loss = []
        time_input = []

        loss_fn = torch.nn.MSELoss()

        target_tensor = torch.ones((batch_size, output_size)).to(DEVICE_NAME).to(DTYPE)
        input_tensor = torch.ones((batch_size, input_size)).to(DEVICE_NAME).to(DTYPE)
        output_ref = (
            torch.ones((batch_size, output_size)).to(DEVICE_NAME).to(DTYPE) * 0.1074
        )
        for i in range(N_ITERS):

            output_tensor = model_torch(input_tensor)

            timer_loss = time.perf_counter()
            loss = loss_fn(output_tensor, target_tensor)
            time_loss.append(time.perf_counter() - timer_loss)

            # loss.requires_grad = True
            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

            time_for_input = np.sum(np.array(time_input))
            time_for_loss = np.sum(np.array(time_loss))
            elapsed_time = (
                time.perf_counter() - timer_start - time_for_input - time_for_loss
            )

            throughput = batch_size / elapsed_time
            time_input = []
            time_loss = []
            timer_start = time.perf_counter()
            if i > WARMUP_ITERS:
                throughputs.append(throughput)
                elapsed_times.append(elapsed_time)
            if debug:
                print(
                    f"Iteration#{i}: time={int(elapsed_time * 1000000)}[µs] thp={throughput}/s"
                )

        mean_training_throughput = np.mean(throughputs[1:])

        # Append the batch size and throughput results to the result structure
        bench_result["training_throughputs"].append(mean_training_throughput)
        bench_result["training_time"].append(N_ITERS * (np.mean(elapsed_times)))

        print(f"Elapsed times: {np.mean(elapsed_times)}")
        print(f"Time for {N_ITERS} training: {N_ITERS*(np.mean(elapsed_times) )}[s]")
        print(
            f"Finished training benchmark. Mean throughput is {mean_training_throughput}/s. Waiting 5s for GPU to cool down."
        )
        time.sleep(5)

        # Inference
        model.eval()

        if OPTIMISE_MODEL:
            model_torch = ipex.optimize(model)
        else:
            model_torch = model

        throughputs = []
        elapsed_times = []

        with torch.no_grad():
            input_tensor = torch.randn((batch_size, input_size)).to(DEVICE_NAME)

        timer_start = time.perf_counter()
        for i in range(N_ITERS):
            with torch.no_grad():
                output_tensor = model_torch(input_tensor)

            elapsed_time = time.perf_counter() - timer_start
            throughput = batch_size / elapsed_time

            if i > WARMUP_ITERS:
                elapsed_times.append(elapsed_time)
                throughputs.append(throughput)
            if debug:
                print(
                    f"Iteration#{i}: time={int(elapsed_time * 1000000)}[µs] thp={throughput}/s"
                )

            time_loss = []
            timer_start = time.perf_counter()

        mean_inference_throughput = np.mean(throughputs[1:])
        bench_result["inference_throughputs"].append(mean_inference_throughput)
        bench_result["inference_time"].append(
            (N_ITERS - WARMUP_ITERS) * (np.mean(elapsed_times))
        )

        print(f"Elapsed times per iter: {np.mean(elapsed_times)}")
        print(f"Time for {N_ITERS} inference: {N_ITERS*(np.mean(elapsed_times))}[s]")

        print(
            f"Finished inference benchmark. Mean throughput is {mean_inference_throughput}/s. Waiting 10s for GPU to cool down."
        )

    # Print results in the desired tab-separated format
    print(f"Mean throughput: ")
    for i in range(len(bench_result["batch_sizes"])):
        print(
            f"{bench_result['training_throughputs'][i]:.1e}, {bench_result['inference_throughputs'][i]:.1e}"
        )
    if path:
        with open(path, "w") as json_file:
            json.dump(
                bench_result, json_file, indent=4
            )  # Use indent for pretty-printing


def test_use_cases(width):
    print("Benchmark")
    input_size = width
    hidden_sizes = [width] * 11
    output_size = width
    batch_sizes = [2**17]
    start_training(input_size, hidden_sizes, output_size, batch_sizes)

    # Image compression
    print("Image compression")
    input_size = width
    hidden_sizes = [width] * 2
    output_size = 1
    batch_sizes = [2**22]
    start_training(input_size, hidden_sizes, output_size, batch_sizes)

    # NeRF
    print("Nerf")
    input_size = width
    hidden_sizes = [width] * 4
    output_size = 4
    batch_sizes = [2**20]
    start_training(input_size, hidden_sizes, output_size, batch_sizes)

    # Pinns
    print("Pinns ")
    input_size = 3
    hidden_sizes = [width] * 5
    output_size = 3
    batch_sizes = [
        2**17,
    ]
    start_training(input_size, hidden_sizes, output_size, batch_sizes)


if __name__ == "__main__":
    # Benchmark
    WIDTHS = [16, 32, 64, 128]
    for WIDTH in WIDTHS:
        print(f"WIDTH: {WIDTH}")
        input_size = WIDTH
        hidden_sizes = [WIDTH] * 4
        output_size = WIDTH
        batch_sizes = [
            2**10,
            2**11,
            2**12,
            2**13,
            2**14,
            2**15,
            2**16,
            2**17,
            2**18,
            2**19,
            2**20,
            2**21,
        ]
        # Test benchmark
        start_training(
            input_size,
            hidden_sizes,
            output_size,
            batch_sizes,
            f"../benchmarks/results/bench_result_pytorch_{DEVICE_NAME}_width{WIDTH}.json",
        )

        # Test Uses cases
        test_use_cases(WIDTH)
