#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

#include <Eigen/Core>

int transform_2d_index(int x, int y, int len_y) { return x * len_y + y; }
int transform_3d_index(int x, int y, int z, int len_y, int len_z) {
  return x * len_y * len_z + y * len_z + z;
}

__global__ void
tile_matmul_kernel(torch::PackedTensorAccessor32<float, 3> tensor, int limit_x,
                   int limit_y, int limit_z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_x && j < limit_y && k < limit_z) {
    // at::Tensor mat_a_row_tiles = mat_a_tiles.index({Slice(), i, Slice()});
    // at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[j].item<int>()];

    // at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[j][k].item<int>()];
    // partial_sum[k] += at::matmul(tile_a, tile_b).squeeze();
    // The following two are to be done externally...
    // result.index_put_({i, Slice()}, partial_sum.flatten().index({Slice(0,
    // mat_b_shape[1])})); partial_sum = partial_sum.zero_();

    Eigen::Matrix<float, 3, 3> t1;
    t1.setZero();
    t1(1, 1) = 5.0f;
    // printf("%f", t1(0, 0));
    // printf("%f\n", t1(1, 1));
    tensor[i][j][k] = 1.0f;
  }
}

int ceil_int_div(int a, int b) { return (a + b - 1) / b; }

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2]) {
  if (at::cuda::is_available()) {
    mat_a_tiles.to(torch::Device("cuda:0"));
    mat_a_tiles_map.to(torch::Device("cuda:0"));
    mat_b_tiles.to(torch::Device("cuda:0"));
    mat_b_tiles_map.to(torch::Device("cuda:0"));
    std::cout << "----------------------------------" << std::endl;
    // Using std and at namespaces
    int i = 4;
    int j = 5;
    int k = 1;
    at::Tensor mat_a_row_tiles = mat_a_tiles.index(
        {torch::indexing::Slice(), i, torch::indexing::Slice()});
    at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[j].item<int>()];
    at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[j][k].item<int>()];
    at::Tensor res = at::matmul(tile_a, tile_b);
    std::cout << res << std::endl;
    // Using a CUDA-compatible apporach
    const int64_t *mat_a_tiles_shape = mat_a_tiles.sizes().data();
    const int64_t *mat_b_tiles_shape = mat_b_tiles.sizes().data();
    auto mat_a_tiles_accessor = mat_a_tiles.data_ptr<float>();
    auto mat_b_tiles_accessor = mat_b_tiles.data_ptr<float>();
    auto mat_a_tiles_map_accessor = mat_a_tiles_map.accessor<float, 1>();
    auto mat_b_tiles_map_accessor = mat_b_tiles_map.accessor<float, 2>();
    Eigen::MatrixXf tile_a_ = Eigen::Map<Eigen::MatrixXf>(
        &mat_a_tiles_accessor[transform_3d_index(mat_a_tiles_map_accessor[j], i,
                                                 0, mat_a_tiles_shape[1],
                                                 mat_a_tiles_shape[2])],
        1, mat_a_tiles_shape[2]);
    Eigen::MatrixXf tile_b_ = Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
                                         Eigen::Stride<1, Eigen::Dynamic>>(
        &mat_b_tiles_accessor[transform_3d_index(mat_b_tiles_map_accessor[j][k],
                                                 0, 0, mat_b_tiles_shape[1],
                                                 mat_b_tiles_shape[2])],
        mat_b_tiles_shape[1], mat_b_tiles_shape[2],
        Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
    Eigen::VectorXf res_ = (tile_a_ * tile_b_).transpose();
    std::cout << res_ << std::endl;
    std::cout << "----------------------------------" << std::endl;
    return mat_a_tiles;
    // cudaDeviceProp prop; outer_stride
    // cudaGetDeviceProperties(&prop, 0);
    // printf("%s\n", prop.name);
    // int *max_threads_dim = prop.maxThreadsDim;
    // // Debugging- TEMP
    // int dim_a = 1024;
    // int dim_b = 1024;
    // int dim_c = 62;
    // mat_b_tiles = at::zeros({dim_a, dim_b,
    // dim_c}).to(torch::Device("cuda:0")); if (max_threads_dim[0] >= dim_a &&
    // max_threads_dim[1] >= dim_b &&
    //     max_threads_dim[2] >= dim_c) {
    //   // If multiple blocks are not required
    //   dim3 grid(dim_a, dim_b, dim_c);
    //   dim3 block(1, 1, 1);
    //   printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    //          grid.x, grid.y, grid.z, block.x, block.y, block.z);
    //   tile_matmul_kernel<<<grid, block>>>(
    //       mat_b_tiles.packed_accessor32<float, 3>(), dim_a, dim_b, dim_c);
    // } else {
    //   // If multiple blocks are required
    //   dim3 grid(max_threads_dim[0], max_threads_dim[1], max_threads_dim[2]);
    //   dim3 block(ceil_int_div(dim_a, max_threads_dim[0]),
    //              ceil_int_div(dim_b, max_threads_dim[1]),
    //              ceil_int_div(dim_c, max_threads_dim[2]));
    //   printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    //          grid.x, grid.y, grid.z, block.x, block.y, block.z);
    //   tile_matmul_kernel<<<grid, block>>>(
    //       mat_b_tiles.packed_accessor32<float, 3>(), dim_a, dim_b, dim_c);
  }
  cudaDeviceSynchronize();
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  std::cout << mat_b_tiles.amin().item<float>()
            << std::endl; // To validate logic
  // }
  return mat_a_tiles;
}