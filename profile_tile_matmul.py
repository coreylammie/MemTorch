import time

import torch

import memtorch
import memtorch_bindings
import memtorch_cuda_bindings
from memtorch.bh.crossbar.Tile import gen_tiles

tile_shape = (18, 16)
test_shape_a = (200, 255)
test_shape_b = (255, 255)
a = torch.zeros(test_shape_a).uniform_(0, 1)
b = torch.zeros(test_shape_b).uniform_(0, 1)
tile_a_tiles, tile_a_map = gen_tiles(a, tile_shape, input=True)
tile_b_tiles, tile_b_map = gen_tiles(b, tile_shape, input=False)
ADC_resolution = 4
overflow_rate = 0.0

# Without quantization
print("----------------------------")
print("Without quantization")
print("----------------------------")
start_time = time.time()
python_res = memtorch.bh.crossbar.tile_matmul(
    tile_a_tiles, tile_a_map, test_shape_a, tile_b_tiles, tile_b_map, test_shape_b
)
elapsed_time = time.time() - start_time
print("Pure Python")
print(python_res)
print(python_res.shape)
print(elapsed_time)
start_time = time.time()
cpp_res = memtorch_bindings.tile_matmul(
    tile_a_tiles, tile_a_map, test_shape_a, tile_b_tiles, tile_b_map, test_shape_b
)
elapsed_time = time.time() - start_time
print("C++ (CPU)")
print(cpp_res)
print(cpp_res.shape)
print(elapsed_time)
start_time = time.time()
cuda_res = memtorch_cuda_bindings.tile_matmul(
    tile_a_tiles, tile_a_map, test_shape_a, tile_b_tiles, tile_b_map, test_shape_b
)
elapsed_time = time.time() - start_time
print("CUDA (GPU)")
print(cuda_res)
print(cuda_res.shape)
print(elapsed_time)

# With quantization
print("----------------------------")
print("With quantization")
print("----------------------------")
start_time = time.time()
python_res = memtorch.bh.crossbar.tile_matmul(
    tile_a_tiles,
    tile_a_map,
    test_shape_a,
    tile_b_tiles,
    tile_b_map,
    test_shape_b,
    ADC_resolution=ADC_resolution,
    ADC_overflow_rate=overflow_rate,
    quant_method="linear",
)
elapsed_time = time.time() - start_time
print("Pure Python")
print(python_res)
print(python_res.shape)
print(elapsed_time)
start_time = time.time()
cpp_res = memtorch_bindings.tile_matmul(
    tile_a_tiles,
    tile_a_map,
    test_shape_a,
    tile_b_tiles,
    tile_b_map,
    test_shape_b,
    ADC_resolution,
    overflow_rate,
    0,
)
elapsed_time = time.time() - start_time
print("C++ (CPU)")
print(cpp_res)
print(cpp_res.shape)
print(elapsed_time)
start_time = time.time()
cuda_res = memtorch_cuda_bindings.tile_matmul(
    tile_a_tiles,
    tile_a_map,
    test_shape_a,
    tile_b_tiles,
    tile_b_map,
    test_shape_b,
    ADC_resolution,
    overflow_rate,
    0,
)
elapsed_time = time.time() - start_time
print("CUDA (GPU)")
print(cuda_res)
print(cuda_res.shape)
print(elapsed_time)
