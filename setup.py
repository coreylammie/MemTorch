import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import include_paths

version = "1.1.3"
CUDA = False


def create_version_py(version, CUDA):
    file = open("memtorch/version.py", "w")
    if CUDA:
        version_string = "__version__ = '{}'".format(version)
    else:
        version_string = "__version__ = '{}-cpu'".format(version)

    file.write(version_string)
    file.close()


create_version_py(version, CUDA)
if CUDA:
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="memtorch_cuda_bindings",
            sources=glob.glob("memtorch/cu/*.cu")
            + glob.glob("memtorch/cu/*.cpp")
            + ["memtorch/cpp/gen_tiles.cpp"],
            library_dirs=["memtorch/submodules"],
            include_dirs=[
                os.path.join(os.getcwd(), relative_path)
                for relative_path in ["memtorch/cu/", "memtorch/submodules/eigen/"]
            ],
        ),
        CppExtension(
            name="memtorch_bindings",
            sources=glob.glob("memtorch/cpp/*.cpp"),
            include_dirs=["memtorch/cpp/"],
        ),
    ]
    name = "memtorch"
else:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    ext_modules = [
        CppExtension(
            name="memtorch_bindings",
            sources=glob.glob("memtorch/cpp/*.cpp"),
            include_dirs=["memtorch/cpp/"],
        )
    ]
    name = "memtorch-cpu"

if __name__ == "__main__":
    setup(
        name=name,
        version=version,
        description="A Simulation Framework for Memristive Deep Learning Systems",
        long_description="A Simulation Framework for Memristive Deep Learning Systems which integrates directly with the well-known PyTorch Machine Learning (ML) library",
        url="https://github.com/coreylammie/MemTorch",
        license="GPL-3.0",
        author="Corey Lammie",
        author_email="coreylammie@jcu.edu.au",
        setup_requires=["ninja"],
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        packages=find_packages(),
        install_requires=[
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "torch>=1.2.0",
            "torchvision",
            "matplotlib",
            "seaborn",
            "ipython",
            "lmfit",
        ],
        include_package_data=True,
        python_requires=">=3.6",
    )
