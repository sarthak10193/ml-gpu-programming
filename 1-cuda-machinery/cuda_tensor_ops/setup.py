import os
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

version = os.environ.get("cuda_machinary", "0.1.0")

# Get directory of setup.py
this_dir = Path(__file__).parent.absolute()

# Determine if we should use Ninja
use_ninja = os.getenv("USE_NINJA", "ON").upper() == "ON"

# Detect CUDA version
def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in output.split("\n") if "release" in line][0]
        version = version_line.split("release ")[1].split(",")[0]
        return version
    except:
        return None

cuda_version = get_cuda_version()
print(f"CUDA version: {cuda_version}")

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="cuda_machinary",
        sources=[
            "csrc/tensor_ops_api.cpp",
            "csrc/tensor_ops/linear.cpp",
            "csrc/tensor_ops/linear_kernel.cu",
        ],
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_80,code=sm_80",
                "--use_fast_math",
            ],
        },
    )

)

setup(
    name="cuda_machinary",
    version=version,
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
    ],
    author="Sarthak Arora",
    description="CUDA + CPP + Pytorch + Python Stack Example",
    keywords="cuda, deep learning, pytorch, tensor operations",
)