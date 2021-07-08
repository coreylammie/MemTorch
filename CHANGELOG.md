## Added

1. C++ and CUDA bindings for `memtorch.bh.crossbar.Tile.tile_matmul`.

Using an NVIDIA GeForce GTX 1080, a tile shape of (25, 25), and two tensors of size (500, 500), the runtime of `tile_matmul` without quantization support is reduced by 2.45x and 5.48x, for CPU-bound and GPU-bound operation, respectively. With an ADC resolution of 4 bits and an overflow rate of 0.0, the runtime of `tile_matmul` with quantization support is reduced by 2.30x and 105.27x, for CPU-bound and GPU-bound operation, respectively.

| Implementation         | Runtime Without Quantization Support (s) | Runtime With Quantization Support (s) |
| ---------------------- | ---------------------------------------- | ------------------------------------- |
| Pure Python (Previous) | 6.917784                                 | 27.099764                             |
| C++ (CPU-bound)        | 2.822265                                 | 11.736974                             |
| CUDA (GPU-bound)       | 1.262861                                 | 0.2574267                             |

3. `Eigen` integration with C++ and CUDA bindings.
4. Additional unit tests.

## Enhanced

1. Modularized C++ and CUDA `quantize` bindings.
2. Enhanced functionality of `naive_progam` and added additional input arguments to dictate logic for stuck devices.

## Fixed

1. Removed debugging code from `naive_progam`.
