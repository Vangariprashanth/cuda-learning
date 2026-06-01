"""
setup.py — Build the CUDA FPS Extension
=========================================

HOW PYTORCH CUDA EXTENSIONS WORK:
    PyTorch ships with torch.utils.cpp_extension which wraps nvcc
    and handles all the include paths, library links, and ABI flags.
    You don't need a separate CMakeLists.txt — setuptools does it all.

BUILD COMMAND:
    pip install -e .          (editable install — recommended for development)
    python setup.py build_ext --inplace   (build .so in current directory)

WHAT GETS BUILT:
    fps_cuda_ext.cpython-3xx-linux-gnu.so
    This .so is importable as: import fps_cuda_ext

VERIFY BUILD:
    python -c "import fps_cuda_ext; print('OK')"

COMMON ERRORS:
    - "nvcc not found": add /usr/local/cuda/bin to PATH
    - "CUDA version mismatch": ensure nvcc and PyTorch CUDA versions match
      Check: nvcc --version  vs  python -c "import torch; print(torch.version.cuda)"
    - "undefined symbol": ABI mismatch — rebuild with same Python/PyTorch version
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Find CUDA include path (usually /usr/local/cuda/include)
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

setup(
    name='fps_cuda_ext',
    version='1.0.0',
    description='CUDA FPS extension for learning and profiling',
    ext_modules=[
        CUDAExtension(
            name='fps_cuda_ext',
            sources=[
                'src/fps_bindings.cpp',
                'src/fps_cuda.cu',
            ],
            extra_compile_args={
                # nvcc flags
                'nvcc': [
                    '-O2',
                    '-arch=sm_80',           # change to match your GPU:
                                             # sm_75 = Turing (RTX 2080)
                                             # sm_86 = Ampere (RTX 3090)
                                             # sm_89 = Ada (RTX 4090)
                                             # sm_80 = A100
                    '--ptxas-options=-v',    # print register usage — important for occupancy analysis
                    '-lineinfo',             # line number info for Nsight
                    '--use_fast_math',       # faster math (rsqrtf, etc.) — OK for our use
                ],
                # Host C++ flags
                'cxx': [
                    '-O2',
                    '-std=c++17',
                ],
            },
            include_dirs=[
                os.path.join(cuda_home, 'include'),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
)
