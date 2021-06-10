#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

void tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                 int mat_a_shape[2], at::Tensor mat_b_tiles,
                 at::Tensor mat_b_tiles_map, int mat_b_shape[2]) {
  at::Tensor mat_b_n_dim = at::_shape_as_tensor(mat_b_tiles);
  int tile_shape[2] = {mat_b_n_dim[-2].item<int>(),
                       mat_b_n_dim[-1].item<int>()};
  std::cout << tile_shape[0] << std::endl;
  std::cout << tile_shape[1] << std::endl;
  return;
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
          tile_matmul(mat_a_tiles, mat_a_tiles_map, mat_a_shape_array,
                      mat_b_tiles, mat_b_tiles_map, mat_b_shape_array);
        });
}