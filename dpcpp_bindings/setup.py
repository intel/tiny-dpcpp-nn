import hashlib
import os
import shlex
import shutil
import sys
from pathlib import Path

from setuptools import setup

import torch.utils.cpp_extension as torch_cpp_extension
from torch.utils.cpp_extension import BuildExtension, SyclExtension


def _wrap_sycl_host_flags_icpx_with_unnamed_lambda(cflags):
    """Compile as a single icpx invocation (no host-compiler split).

    PyTorch's default SYCL extension build uses ``-fsycl-host-compiler``, which
    breaks oneDPL device policies and unnamed SYCL kernels in this project.
    """
    return [shlex.quote(f) for f in cflags + ["-fsycl-unnamed-lambda"]]


torch_cpp_extension._wrap_sycl_host_flags = (
    _wrap_sycl_host_flags_icpx_with_unnamed_lambda
)

_orig_library_paths = torch_cpp_extension.library_paths


def _library_paths_pytorch_xpu_sycl(
    device_type="cpu", torch_include_dirs=True, cross_target_platform=None
):
    paths = _orig_library_paths(
        device_type, torch_include_dirs, cross_target_platform
    )
    if device_type != "xpu":
        return paths
    venv_lib = os.path.join(sys.prefix, "lib")
    sycl_home = torch_cpp_extension.SYCL_HOME
    if sycl_home is not None:
        compiler_lib = os.path.realpath(os.path.join(sycl_home, "lib"))
        paths = [p for p in paths if os.path.realpath(p) != compiler_lib]
    return [venv_lib] + paths


torch_cpp_extension.library_paths = _library_paths_pytorch_xpu_sycl


class FilterVenvSyclIncludesBuildExtension(BuildExtension):
    """Drop venv ``include/`` from the compiler search path.

    ``intel-sycl-rt`` (pulled in by torch xpu) installs SYCL 2026 headers under
    ``$VIRTUAL_ENV/include``. Those must not take precedence over the oneAPI
    compiler headers matching ``icpx``.
    """

    def build_extension(self, ext):
        venv_include = os.path.realpath(os.path.join(sys.prefix, "include"))
        self.compiler.include_dirs = [
            d
            for d in self.compiler.include_dirs
            if os.path.realpath(d) != venv_include
        ]
        return super().build_extension(ext)

# Here ze_loader is not necessary, just used to check libraries linker
# libraries = ["ze_loader"] if IS_LINUX else []

_BINDINGS_DIR = Path(__file__).resolve().parent
_SYCL_SOURCE_DIR = _BINDINGS_DIR / "_sycl_sources"


def _as_sycl_sources(cpp_sources: list[str]) -> list[str]:
    """Expose .cpp sources as .sycl so PyTorch builds them with icpx."""
    _SYCL_SOURCE_DIR.mkdir(exist_ok=True)
    sycl_sources = []
    for source in cpp_sources:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (_BINDINGS_DIR / source_path).resolve()
        if source_path.suffix != ".cpp":
            sycl_sources.append(str(source_path))
            continue
        digest = hashlib.md5(str(source_path).encode()).hexdigest()[:8]
        link_path = _SYCL_SOURCE_DIR / f"{source_path.stem}_{digest}.sycl"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(source_path)
        sycl_sources.append(str(link_path))
    return sycl_sources


def _default_oneapi_root() -> str:
    for candidate in (
        os.path.expanduser("~/intel/oneapi/2026.1"),
        "/opt/intel/oneapi",
    ):
        if os.path.isdir(candidate):
            return os.path.realpath(candidate)
    return "/opt/intel/oneapi"


def _resolve_compiler_root(oneapi_install: str | None) -> str | None:
    if oneapi_install is None:
        return None
    candidates = [
        os.path.join(oneapi_install, "compiler", "latest"),
        os.path.join(oneapi_install, "opt", "compiler"),
        oneapi_install,
    ]
    for candidate in candidates:
        sycl_include = os.path.join(candidate, "include", "sycl")
        if os.path.isdir(sycl_include):
            return os.path.realpath(candidate)
    return None


oneapi_root = os.getenv("ONEAPI_ROOT", _default_oneapi_root())
if not os.path.isdir(oneapi_root):
    oneapi_root = None

dpcpp_path = os.getenv("CMPLR_ROOT")
if dpcpp_path is None:
    dpcpp_path = _resolve_compiler_root(oneapi_root)

if dpcpp_path is None:
    raise RuntimeError(
        "CMPLR_ROOT or ONEAPI_ROOT must point to an Intel oneAPI installation"
    )

if shutil.which("icpx") and os.environ.get("CXX") in (None, "c++", "g++"):
    os.environ["CXX"] = "icpx"

dpcpp_sycl_path = os.path.join(dpcpp_path, "include", "sycl")

oneapi_lib_dirs = []
oneapi_include_dirs = []
if oneapi_root is not None:
    mkl_lib = os.path.join(oneapi_root, "mkl", "latest", "lib")
    if os.path.isdir(mkl_lib):
        oneapi_lib_dirs.append(mkl_lib)
    elif os.path.isdir(os.path.join(oneapi_root, "lib")):
        oneapi_lib_dirs.append(os.path.join(oneapi_root, "lib"))
    include_candidates = [os.path.join(oneapi_root, "include")]
    include_candidates.extend(
        os.path.join(oneapi_root, component, "latest", "include")
        for component in ("dpl", "mkl")
    )
    for include_dir in include_candidates:
        if os.path.isdir(include_dir) and include_dir not in oneapi_include_dirs:
            oneapi_include_dirs.append(include_dir)

_venv_lib_dir = os.path.join(sys.prefix, "lib")
_mkl_link_libs = [
    os.path.join(_venv_lib_dir, name)
    for name in (
        "libmkl_sycl_rng.so.6",
        "libmkl_intel_lp64.so.3",
        "libmkl_gnu_thread.so.3",
        "libmkl_core.so.3",
    )
]
_extra_link_args = [f"-Wl,-rpath,{_venv_lib_dir}", *_mkl_link_libs]

libraries = []

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

# Limit offline device codegen to the selected GPU family when not set explicitly.
if os.getenv("TORCH_XPU_ARCH_LIST") is None:
    arch_by_target = {"PVC": "pvc", "BMG": "bmg", "ACM": "dg2"}
    os.environ["TORCH_XPU_ARCH_LIST"] = arch_by_target.get(
        os.getenv("TARGET_DEVICE", "BMG").upper(), "bmg"
    )
    print(
        "Info: TORCH_XPU_ARCH_LIST is not set, defaulting to "
        f"{os.environ['TORCH_XPU_ARCH_LIST']}"
    )

cpp_sources = [
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
]

sycl_compile_flags = [
    f"-DTARGET_DEVICE={target_device}",
    "-std=c++20",
    "-fPIC",
    "-DSYCL2020_CONFORMANT_APIS",
    "-fp-model=precise",
    "-fsycl-device-code-split=per_kernel",
]

setup(
    name="tiny_dpcpp_nn",
    version="0.0.1",
    description="Python bindings for the tiny-dpcpp-nn library",
    packages=["tiny_dpcpp_nn"],
    ext_modules=[
        SyclExtension(
            "tiny_dpcpp_nn.tiny_dpcpp_nn_pybind_module",
            _as_sycl_sources(cpp_sources),
            libraries=libraries,
            library_dirs=[_venv_lib_dir],
            extra_link_args=_extra_link_args,
            extra_compile_args={
                "cxx": [f"-DTARGET_DEVICE={target_device}", "-std=c++20", "-fPIC"],
                "sycl": sycl_compile_flags,
            },
            include_dirs=(
                ([dpcpp_sycl_path]
                if conda_sycl_path is None
                else [conda_sycl_path])
                + oneapi_include_dirs
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
    cmdclass={"build_ext": FilterVenvSyclIncludesBuildExtension.with_options(use_ninja=True)},
)