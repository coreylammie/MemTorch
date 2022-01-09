#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>
using namespace torch::indexing;

std::tuple<at::Tensor, at::Tensor>
gen_tiles(at::Tensor tensor, int tile_shape[2], bool input,
          torch::TensorOptions tensor_options) {
  c10::IntArrayRef tensor_shape = tensor.sizes();
  int tile_columns;
  int column_start;
  int column_end;
  at::Tensor tiles_map;
  at::Tensor tiles;
  if (input) {
    tile_columns = ceil((float)tensor_shape[1] / tile_shape[0]);
    tiles = at::zeros({tile_columns, tensor_shape[0], tile_shape[0]},
                      tensor_options);
    tiles_map = at::zeros({tile_columns}, tensor_options);
#pragma omp parallel for
    for (int i = 0; i < tile_columns; i++) {
      column_start = i * tile_shape[0];
      if (i == tile_columns - 1) {
        column_end = tensor_shape[1];
      } else {
        column_end = (i + 1) * tile_shape[0];
      }
      tiles.index_put_(
          {i, Slice(0, tensor_shape[0]), Slice(0, column_end - column_start)},
          tensor.index({Slice(), Slice(column_start, column_end)}));
      tiles_map[i] = i;
    }
  } else {
    int tile_rows = ceil((float)tensor_shape[0] / tile_shape[0]);
    tile_columns = ceil((float)tensor_shape[1] / tile_shape[1]);
    tiles_map = at::zeros({tile_rows, tile_columns}, tensor_options);
    int row_start;
    int row_end;
    tiles = at::zeros({tile_rows * tile_columns, tile_shape[0], tile_shape[1]},
                      tensor_options);
#pragma omp parallel for
    for (int i = 0; i < tile_rows; i++) {
      row_start = i * tile_shape[0];
      if (i == tile_rows - 1) {
        row_end = tensor_shape[0];
      } else {
        row_end = (i + 1) * tile_shape[0];
      }
      for (int j = 0; j < tile_columns; j++) {
        column_start = j * tile_shape[1];
        if (j == tile_columns - 1) {
          column_end = tensor_shape[1];
        } else {
          column_end = (j + 1) * tile_shape[1];
        }
        tiles.index_put_({i * tile_columns + j, Slice(0, row_end - row_start),
                          Slice(0, column_end - column_start)},
                         tensor.index({Slice(row_start, row_end),
                                       Slice(column_start, column_end)}));
        tiles_map.index_put_({i, j}, i * tile_columns + j);
      }
    }
  }
  return std::tuple<at::Tensor, at::Tensor>{tiles, tiles_map};
}

void gen_tiles_bindings(py::module_ &m) {
  m.def(
      "gen_tiles",
      [](at::Tensor tensor, std::tuple<int, float> tile_shape, bool input) {
        assert((std::tuple_size<int, float>(tile_shape) == 2));
        int tile_shape_array[2] = {(int)std::get<0>(tile_shape),
                                   (int)std::get<1>(tile_shape)};
        return gen_tiles(tensor, tile_shape_array, input,
                         torch::TensorOptions().device(torch::kCPU));
      },
      py::arg("tensor"), py::arg("tile_shape"), py::arg("input") = false);
}

void gen_tiles_bindings_gpu(py::module_ &m) {
  m.def(
      "gen_tiles",
      [](at::Tensor tensor, std::tuple<int, float> tile_shape, bool input) {
        assert((std::tuple_size<int, float>(tile_shape) == 2));
        int tile_shape_array[2] = {(int)std::get<0>(tile_shape),
                                   (int)std::get<1>(tile_shape)};
        return gen_tiles(tensor, tile_shape_array, input,
                         torch::TensorOptions().device(torch::kCUDA, 0));
      },
      py::arg("tensor"), py::arg("tile_shape"), py::arg("input") = false);
}