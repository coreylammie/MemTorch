#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "quantize.h"
using namespace torch::indexing;

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2]) {
  int mat_a_rows = mat_a_tiles.sizes().end()[-2];
  c10::IntArrayRef mat_b_tiles_shape = mat_b_tiles.sizes();
  c10::IntArrayRef mat_b_tiles_map_shape = mat_b_tiles_map.sizes();
  at::Tensor partial_sum =
      at::zeros({mat_b_tiles_map_shape[1], mat_b_tiles_shape.back()});
  at::Tensor result = at::zeros({mat_a_shape[0], mat_b_shape[1]});
#pragma omp parallel for
  for (int i = 0; i < mat_a_rows; i++) {
    at::Tensor mat_a_row_tiles = mat_a_tiles.index({Slice(), i, Slice()});
    for (int j = 0; j < mat_b_tiles_map_shape[0]; j++) {
      at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[j].item<int>()];
      for (int k = 0; k < mat_b_tiles_map_shape[1]; k++) {
        at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[j][k].item<int>()];
        partial_sum[k] += at::matmul(tile_a, tile_b).squeeze();
      }
      result.index_put_({i, Slice()}, result.index({i, Slice()}) +
                                          partial_sum.flatten().index(
                                              {Slice(0, mat_b_shape[1])}));
      partial_sum = partial_sum.zero_();
    }
  }
  return result;
}

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2],
                       int ADC_resolution, float ADC_overflow_rate,
                       int quant_method) {
  int mat_a_rows = mat_a_tiles.sizes().end()[-2];
  c10::IntArrayRef mat_b_tiles_shape = mat_b_tiles.sizes();
  c10::IntArrayRef mat_b_tiles_map_shape = mat_b_tiles_map.sizes();
  at::Tensor partial_sum =
      at::zeros({mat_b_tiles_map_shape[1], mat_b_tiles_shape.back()});
  at::Tensor result = at::zeros({mat_a_shape[0], mat_b_shape[1]});
#pragma omp parallel for
  for (int i = 0; i < mat_a_rows; i++) {
    at::Tensor mat_a_row_tiles = mat_a_tiles.index({Slice(), i, Slice()});
    for (int j = 0; j < mat_b_tiles_map_shape[0]; j++) {
      partial_sum =
          at::zeros({mat_b_tiles_map_shape[1], mat_b_tiles_shape.back()});
      at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[j].item<int>()];
      for (int k = 0; k < mat_b_tiles_map_shape[1]; k++) {
        at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[j][k].item<int>()];
        at::Tensor result = at::matmul(tile_a, tile_b).squeeze();
        quantize(result, ADC_resolution, ADC_overflow_rate, quant_method);
        partial_sum[k] += result;
      }
      partial_sum = partial_sum.flatten().index({Slice(0, mat_b_shape[1])});
      result.index_put_({i, Slice()}, result.index({i, Slice()}) + partial_sum);
    }
  }
  return result;
}

void tile_matmul_bindings(py::module_ &m) {
  // Binding without quantization support
  m.def(
      "tile_matmul",
      [](at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
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
                           mat_b_tiles, mat_b_tiles_map, mat_b_shape_array);
      },
      py::arg("mat_a_tiles"), py::arg("mat_a_tiles_map"),
      py::arg("mat_a_shape"), py::arg("mat_b_tiles"),
      py::arg("mat_b_tiles_map"), py::arg("mat_b_shape"),
      py::arg("cuda_malloc_heap_size") = NULL);
  // Binding with quantization support
  m.def(
      "tile_matmul",
      [](at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
         std::tuple<int, float> mat_a_shape, at::Tensor mat_b_tiles,
         at::Tensor mat_b_tiles_map, std::tuple<int, float> mat_b_shape,
         int ADC_resolution, float ADC_overflow_rate, int quant_method,
         int cuda_malloc_heap_size) {
        assert((std::tuple_size<int, float>(mat_a_shape) == 2) &&
               (std::tuple_size<int, float>(mat_b_shape) == 2));
        int mat_a_shape_array[2] = {(int)std::get<0>(mat_a_shape),
                                    (int)std::get<1>(mat_a_shape)};
        int mat_b_shape_array[2] = {(int)std::get<0>(mat_b_shape),
                                    (int)std::get<1>(mat_b_shape)};
        return tile_matmul(mat_a_tiles, mat_a_tiles_map, mat_a_shape_array,
                           mat_b_tiles, mat_b_tiles_map, mat_b_shape_array,
                           ADC_resolution, ADC_overflow_rate, quant_method);
      },
      py::arg("mat_a_tiles"), py::arg("mat_a_tiles_map"),
      py::arg("mat_a_shape"), py::arg("mat_b_tiles"),
      py::arg("mat_b_tiles_map"), py::arg("mat_b_shape"),
      py::arg("ADC_resolution"), py::arg("ADC_overflow_rate"),
      py::arg("quant_method"), py::arg("cuda_malloc_heap_size") = NULL);
}