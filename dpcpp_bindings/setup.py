import os
from setuptools import setup

from intel_extension_for_pytorch.xpu.cpp_extension import (
    DPCPPExtension,
    DpcppBuildExtension,
    IS_LINUX,
)

# Here ze_loader is not necessary, just used to check libraries linker
# libraries = ["ze_loader"] if IS_LINUX else []
libraries = []

dpcpp_path = os.getenv("CMPLR_ROOT")
dpcpp_sycl_path = os.path.join(dpcpp_path, "include", "sycl")

conda_path = os.getenv("CONDA_PREFIX")
conda_sycl_path = None
if conda_path is not None:
    conda_sycl_path = os.path.join(conda_path, "include", "sycl")
    if not os.path.exists(conda_sycl_path):
        conda_sycl_path = None


target_device_map = {
    "PVC": "0",
    "BMG": "0",
    "ACM": "1",
}
target_device = target_device_map.get(os.getenv("TARGET_DEVICE", "BMG").upper())
if target_device is None:
    raise ValueError(f"TARGET_DEVICE must be one of {sorted(target_device_map.keys())}")
if os.getenv("TARGET_DEVICE") is None:
    print("Info: TARGET_DEVICE is not set, defaulting to PVC/BMG")
else:
    print(f"Info: TARGET_DEVICE is set to {os.getenv('TARGET_DEVICE')}")

setup(
    name="tiny_dpcpp_nn",
    version="0.0.1",
    description="Python bindings for the tiny-dpcpp-nn library",
    packages=["tiny_dpcpp_nn"],
    ext_modules=[
        DPCPPExtension(
            "tiny_dpcpp_nn.tiny_dpcpp_nn_pybind_module",
            [
                "tiny_dpcpp_nn/pybind_module.cpp",
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1664none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLP.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1664sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1616none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1632sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf16128none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp16128relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp16128none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPbf1664.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1632relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimd.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1616none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPfp1664.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1616relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp16128sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPfp1632.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf16128relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1664relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPbf1632.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPfp16128.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPfp1616.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1664relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1664sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1632sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPbf1616.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1664none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1616relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1616sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1616sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1632none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "SwiftNetMLPbf16128.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdfp1632relu.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf1632none.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "network", "kernel_esimdbf16128sigmoid.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "optimizers", "sgd.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "optimizers", "adam.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "common", "SyclGraph.cpp"),
                os.path.join(os.path.dirname(__file__), "..", "source", "common", "common.cpp"),
            ],
            libraries=libraries,
            extra_compile_args={'cxx': [f'-DTARGET_DEVICE={target_device}', '-std=c++20', '-fPIC']},
            include_dirs=(
                ([dpcpp_sycl_path]
                if conda_sycl_path is None
                else [conda_sycl_path])
                + [
                    os.path.join(os.path.dirname(__file__), "../include"),
                    os.path.join(os.path.dirname(__file__), "../include/network"),
                    os.path.join(os.path.dirname(__file__), "../include/common"),
                    os.path.join(os.path.dirname(__file__), "../include/encodings"),
                    os.path.join(os.path.dirname(__file__), "../include/optimizers"),
                    os.path.join(os.path.dirname(__file__), "../extern/json"),
                    os.path.join(os.path.dirname(__file__), "../extern/pybind11_json"),
                ]
            ),
        )
    ],
    cmdclass={"build_ext": DpcppBuildExtension.with_options(use_ninja=True)},
)