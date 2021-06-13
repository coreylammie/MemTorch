#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

using namespace torch::indexing;

at::Tensor tile_matmul_row(at::Tensor mat_a_row_tiles,
                           at::Tensor mat_a_tiles_map, at::Tensor mat_b_tiles,
                           at::Tensor mat_b_tiles_map, int mat_b_shape[2],
                           at::Tensor partial_sum) {
  partial_sum = partial_sum.zero_();
  at::Tensor mat_b_tiles_map_shape = at::_shape_as_tensor(mat_b_tiles_map);
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
  for (int j = 0; j < mat_b_tiles_map_shape[1].item<int>(); j++) {
    for (int i = 0; i < mat_b_tiles_map_shape[0].item<int>(); i++) {
      at::Tensor tile_a =
          mat_a_row_tiles[mat_a_tiles_map[i].item<int>()]; //.to(torch::kCUDA);
      at::Tensor tile_b =
          mat_b_tiles[mat_b_tiles_map[i][j].item<int>()]; //.to(torch::kCUDA);
      partial_sum[j] += at::matmul(tile_a, tile_b).squeeze();
    }
  }
  return partial_sum.flatten().index({Slice(0, mat_b_shape[1])});
}

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2]) {
  at::Tensor mat_b_n_dim = at::_shape_as_tensor(mat_b_tiles);
  at::Tensor partial_sum =
      at::zeros({mat_b_n_dim[-2].item<int>(),
                 mat_b_n_dim[-1].item<int>()}); // .to(torch::kCUDA);
  if (at::_shape_as_tensor(mat_a_tiles)[-2].item<int>() > 1) {
    at::Tensor result =
        at::zeros({mat_a_shape[0], mat_b_shape[1]}); //.to(torch::kCUDA);
#pragma omp parallel for
    for (int i = 0; i < at::_shape_as_tensor(mat_a_tiles)[-2].item<int>();
         i++) {
      result.index_put_(
          {i, Slice()},
          tile_matmul_row(mat_a_tiles.index({Slice(), i, Slice()}),
                          mat_a_tiles_map, mat_b_tiles, mat_b_tiles_map,
                          mat_b_shape, partial_sum));
    }
    return result;
  } else {
    return tile_matmul_row(mat_a_tiles, mat_a_tiles_map, mat_b_tiles,
                           mat_b_tiles_map, mat_b_shape, partial_sum);
  }
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