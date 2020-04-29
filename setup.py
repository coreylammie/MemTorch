from setuptools import setup, find_packages
import torch


CUDA = False
if CUDA:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
    ext_modules = [
        CUDAExtension('cuda_quantization', [
        'memtorch/cu/quantize/quant_cuda.cpp',
        'memtorch/cu/quantize/quant.cu'
        ], extra_include_paths='memtorch/cu/quantize'),
        CppExtension('quantization', [
        'memtorch/cpp/quantize/quant.cpp'
    ])]
    name = 'memtorch'
else:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
    ext_modules = [
        CppExtension('quantization', [
        'memtorch/cpp/quantize/quant.cpp'
    ])]
    name = 'memtorch-cpu'

setup(name=name,
      version='1.0.1',
      description='A Simulation Framework for Memristive Deep Learning Systems',
      long_description='A Simulation Framework for Memristive Deep Learning Systems which integrates directly with the well-known PyTorch Machine Learning (ML) library',
      url='https://github.com/coreylammie/MemTorch',
      license='GPL-3.0',
      author='Corey Lammie',
      author_email='coreylammie@jcu.edu.au',
      ext_modules=ext_modules,
      cmdclass={
          'build_ext': BuildExtension
      },
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'sklearn',
          'torch>=1.2.0',
          'matplotlib',
          'seaborn'
      ],
      python_requires='>=3.6',
      include_package_data=True
 )
