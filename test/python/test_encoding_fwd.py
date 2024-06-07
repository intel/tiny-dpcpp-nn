import numpy as np
import torch
import tiny_dpcpp_nn as tnn
import json

if __name__ == "__main__":
    filepath = "/nfs/site/home/yuankai/code/tiny-dpcpp-nn/test/tiny-dpcpp-data/ref_values/encoding/grid_image/"
    with open("data/config_hash.json") as config_file:
        config = json.load(config_file)

    encoding_input = np.genfromtxt(f"{filepath}/input_encoding.csv", delimiter=",")
    encoding_params = np.genfromtxt(f"{filepath}/encoding_params.csv", delimiter=",")
    encoding_output_ref = np.genfromtxt(
        f"{filepath}/output_encoding.csv", delimiter=","
    )
    encoding = tnn.Encoding(
        n_input_dims=2,
        encoding_config=config["encoding"],
        dtype=torch.float,
        filepath=f"{filepath}/",
    )
    print("encoding_params: ", encoding_params)
    print("encoding.params: ", encoding.params)
    encoding_output = encoding(
        torch.Tensor(encoding_input).to("xpu").to(dtype=torch.float)
    )

    print("Print fwd: ", encoding_output[0, :])
    print("-------")
    print("Print ref:", encoding_output_ref[0, :])
