#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "gen_tiles.h"
#include "tile_matmul_kernels.cuh"

at::Tensor tiled_inference(at::Tensor input, int input_shape[2],
                           int tile_shape[2], at::Tensor weight_tiles,
                           at::Tensor weight_tiles_map, int weight_shape[2],
                           int cuda_malloc_heap_size) {
  at::Tensor input_tiles;
  at::Tensor input_tiles_map;
  std::tie(input_tiles, input_tiles_map) = gen_tiles(
      input, tile_shape, true, torch::TensorOptions().device(torch::kCUDA, 0));
  return tile_matmul(input_tiles, input_tiles_map, input_shape, weight_tiles,
                     weight_tiles_map, weight_shape, NULL, NULL, -1,
                     cuda_malloc_heap_size);
}

at::Tensor tiled_inference(at::Tensor input, int input_shape[2],
                           int tile_shape[2], at::Tensor weight_tiles,
                           at::Tensor weight_tiles_map, int weight_shape[2],
                           int ADC_resolution, float ADC_overflow_rate,
                           int quant_method, int cuda_malloc_heap_size) {
  at::Tensor input_tiles;
  at::Tensor input_tiles_map;
  std::tie(input_tiles, input_tiles_map) = gen_tiles(
      input, tile_shape, true, torch::TensorOptions().device(torch::kCUDA, 0));
  return tile_matmul(input_tiles, input_tiles_map, input_shape, weight_tiles,
                     weight_tiles_map, weight_shape, ADC_resolution,
                     ADC_overflow_rate, quant_method, cuda_malloc_heap_size);
}

void inference_bindings(py::module_ &m) {
  // Binding without quantization support
  m.def(
      "tiled_inference",
      [](at::Tensor input, std::tuple<int, float> input_shape,
         std::tuple<int, float> tile_shape, at::Tensor weight_tiles,
         at::Tensor weight_tiles_map, std::tuple<int, float> weight_shape,
         int cuda_malloc_heap_size) {
        assert((std::tuple_size<int, float>(input_shape) == 2));
        assert((std::tuple_size<int, float>(tile_shape) == 2));
        assert((std::tuple_size<int, float>(weight_shape) == 3));
        int input_shape_array[2] = {(int)std::get<0>(input_shape),
                                    (int)std::get<1>(input_shape)};
        int tile_shape_array[2] = {(int)std::get<0>(tile_shape),
                                   (int)std::get<1>(tile_shape)};
        int weight_shape_array[2] = {(int)std::get<0>(weight_shape),
                                     (int)std::get<1>(weight_shape)};
        return tiled_inference(input, input_shape_array, tile_shape_array,
                               weight_tiles, weight_tiles_map,
                               weight_shape_array, cuda_malloc_heap_size);
      },
      py::arg("input"), py::arg("input_shape"), py::arg("tile_shape"),
      py::arg("weight_tiles"), py::arg("weight_tiles_map"),
      py::arg("weight_shape"), py::arg("cuda_malloc_heap_size") = 50);
  // Binding with quantization support
  m.def(
      "tiled_inference",
      [](at::Tensor input, std::tuple<int, float> input_shape,
         std::tuple<int, float> tile_shape, at::Tensor weight_tiles,
         at::Tensor weight_tiles_map, std::tuple<int, float> weight_shape,
         int ADC_resolution, float ADC_overflow_rate, int quant_method,
         int cuda_malloc_heap_size) {
        assert((std::tuple_size<int, float>(input_shape) == 2));
        assert((std::tuple_size<int, float>(tile_shape) == 2));
        assert((std::tuple_size<int, float>(weight_shape) == 3));
        int input_shape_array[2] = {(int)std::get<0>(input_shape),
                                    (int)std::get<1>(input_shape)};
        int tile_shape_array[2] = {(int)std::get<0>(tile_shape),
                                   (int)std::get<1>(tile_shape)};
        int weight_shape_array[2] = {(int)std::get<0>(weight_shape),
                                     (int)std::get<1>(weight_shape)};
        return tiled_inference(
            input, input_shape_array, tile_shape_array, weight_tiles,
            weight_tiles_map, weight_shape_array, ADC_resolution,
            ADC_overflow_rate, quant_method, cuda_malloc_heap_size);
      },
      py::arg("input"), py::arg("input_shape"), py::arg("tile_shape"),
      py::arg("weight_tiles"), py::arg("weight_tiles_map"),
      py::arg("weight_shape"), py::arg("ADC_resolution"),
      py::arg("ADC_overflow_rate"), py::arg("quant_method"),
      py::arg("cuda_malloc_heap_size") = 50);
}