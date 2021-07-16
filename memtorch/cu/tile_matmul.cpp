
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#include "tile_matmul_kernels.cuh"

void tile_matmul_bindings(py::module_ &m) {
  m.def(
      "tile_matmul",
      [&](at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
          std::tuple<int, float> mat_a_shape, at::Tensor mat_b_tiles,
          at::Tensor mat_b_tiles_map, std::tuple<int, float> mat_b_shape,
          int cuda_malloc_heap_size) {
        assert((std::tuple_size<int, float>(mat_a_shape) == 2) &&
               (std::tuple_size<int, float>(mat_b_shape) == 2));
        int mat_a_shape_array[2] = {(int)std::get<0>(mat_a_shape),
                                    (int)std::get<1>(mat_a_shape)};
        int mat_b_shape_array[2] = {(int)std::get<0>(mat_b_shape),
                                    (int)std::get<1>(mat_b_shape)};
        return tile_matmul(mat_a_tiles, mat_a_tiles_map, mat_a_shape_array,
                           mat_b_tiles, mat_b_tiles_map, mat_b_shape_array, -1,
                           -1, -1, cuda_malloc_heap_size);
      },
      py::arg("mat_a_tiles"), py::arg("mat_a_tiles_map"),
      py::arg("mat_a_shape"), py::arg("mat_b_tiles"),
      py::arg("mat_b_tiles_map"), py::arg("mat_b_shape"),
      py::arg("cuda_malloc_heap_size") = 50);
  m.def(
      "tile_matmul",
      [&](at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
          std::tuple<int, float> mat_a_shape, at::Tensor mat_b_tiles,
          at::Tensor mat_b_tiles_map, std::tuple<int, float> mat_b_shape,
          int ADC_resolution, float overflow_rate, int quant_method,
          int cuda_malloc_heap_size) {
        assert((std::tuple_size<int, float>(mat_a_shape) == 2) &&
               (std::tuple_size<int, float>(mat_b_shape) == 2));
        int mat_a_shape_array[2] = {(int)std::get<0>(mat_a_shape),
                                    (int)std::get<1>(mat_a_shape)};
        int mat_b_shape_array[2] = {(int)std::get<0>(mat_b_shape),
                                    (int)std::get<1>(mat_b_shape)};
        return tile_matmul(mat_a_tiles, mat_a_tiles_map, mat_a_shape_array,
                           mat_b_tiles, mat_b_tiles_map, mat_b_shape_array,
                           ADC_resolution, overflow_rate, quant_method,
                           cuda_malloc_heap_size);
      },
      py::arg("mat_a_tiles"), py::arg("mat_a_tiles_map"),
      py::arg("mat_a_shape"), py::arg("mat_b_tiles"),
      py::arg("mat_b_tiles_map"), py::arg("mat_b_shape"),
      py::arg("ADC_resolution"), py::arg("ADC_overflow_rate"),
      py::arg("quant_method"), py::arg("cuda_malloc_heap_size") = 50);
}