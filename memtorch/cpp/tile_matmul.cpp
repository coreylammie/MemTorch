#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

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
    // for (int j = 0; j < mat_b_tiles_map_shape[0]; j++) {
    //   at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[j].item<int>()];
    //   for (int k = 0; k < mat_b_tiles_map_shape[1]; k++) {
    //     at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[j][k].item<int>()];
    //     partial_sum[k] += at::matmul(tile_a, tile_b).squeeze();
    //   }
    // }
    for (int j = 0; j < mat_b_tiles_map_shape[1]; j++) {
      for (int k = 0; k < mat_b_tiles_map_shape[0]; k++) {
        at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[k].item<int>()];
        at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[k][j].item<int>()];
        partial_sum = at::matmul(tile_a, tile_b).squeeze().flatten();
        // result[i, j+k] = partial_sum

        // std::cout << partial_sum << std::endl;
        // std::cout << mat_b_tiles_map_shape[1] << std::endl;
        // std::cout << mat_b_tiles_map_shape[0] << std::endl;
        // std::cout << partial_sum << std::endl;
        // std::cout << partial_sum.flatten() << std::endl;
        // return result;
        // std::cout << (j * k) << ", " << (j * k) + j + k << ", "
        //           << partial_sum.numel() << std::endl;
        for (int ii = 0; ii < partial_sum.numel(); ii++) {
          result[i][(j * k) + ii + j] += partial_sum[ii];
        }
        // result.index_put_({i, Slice()}, partial_sum.flatten().index(
        //                                     {Slice(0, mat_b_shape[1])}));
        // // result[i][j] +=
        // //     partial_sum[j]; //.flatten().index({Slice(0,
        // mat_b_shape[1])});
      }
    }
    // return result;
    // std::cout << partial_sum << std::endl;
    // std::cout << partial_sum.flatten() << std::endl;
    // return result;
    // result.index_put_({i, Slice()},
    //                   partial_sum.flatten().index({Slice(0,
    //                   mat_b_shape[1])}));
    // partial_sum = partial_sum.zero_();
  }
  return result;
}

void tile_matmul_bindings(py::module_ &m) {
  m.def("tile_matmul",
        [](at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
           std::tuple<int, float> mat_a_shape, at::Tensor mat_b_tiles,
           at::Tensor mat_b_tiles_map, std::tuple<int, float> mat_b_shape) {
          assert((std::tuple_size<int, float>(mat_a_shape) == 2) &&
                 (std::tuple_size<int, float>(mat_b_shape) == 2));
          int mat_a_shape_array[2] = {(int)std::get<0>(mat_a_shape),
                                      (int)std::get<1>(mat_a_shape)};
          int mat_b_shape_array[2] = {(int)std::get<0>(mat_b_shape),
                                      (int)std::get<1>(mat_b_shape)};
          return tile_matmul(mat_a_tiles, mat_a_tiles_map, mat_a_shape_array,
                             mat_b_tiles, mat_b_tiles_map, mat_b_shape_array);
        });
}