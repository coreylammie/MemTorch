from setuptools import setup, find_packages
import torch


version = '1.1.1'
CUDA = False

def create_version_py(version, CUDA):
    file = open('memtorch/version.py', 'w')
    if CUDA:
        version_string = '__version__ = \'{}\''.format(version)
    else:
        version_string = '__version__ = \'{}-cpu\''.format(version)

    file.write(version_string)
    file.close()

create_version_py(version, CUDA)
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

if __name__ == '__main__':
    setup(name=name,
          version=version,
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
              'torchvision',
              'matplotlib',
              'seaborn',
              'ipython'
          ],
          include_package_data=CUDA,
          python_requires='>=3.6'
     )
