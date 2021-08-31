#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

#include "solve_passive.h"

#include <cs.h>

__global__ void spalloc(cs *ABCD_matrix, int m, int n, int non_zero_elements) {
  // ABCD_matrix = cs_spalloc(m, n, non_zero_elements, 1, 1);
  ABCD_matrix->m = (csi)m;
  ABCD_matrix->n = (csi)n;
  ABCD_matrix->nzmax = (csi)non_zero_elements;
  ABCD_matrix->nz = (csi)non_zero_elements;
  ABCD_matrix->p = (ptrdiff_t *)malloc(sizeof(csi) * non_zero_elements);
  ABCD_matrix->i = (ptrdiff_t *)malloc(sizeof(csi) * non_zero_elements);
  ABCD_matrix->x = (double *)malloc(sizeof(double) * non_zero_elements);
}

__global__ void solve_sparse_linear_alt(cs *ABCD_matrix, double *E_matrix) {
  printf("A.\n");
  cs *ABCD_matrix_compressed = cs_compress((const cs *)ABCD_matrix);
  printf("B.\n");
  // cs_spfree(ABCD_matrix);
  printf("C.\n");
  // auto ok = cs_qrsol(1, ABCD_matrix_compressed, E_matrix);
  printf("D.\n");
  // printf("%d\n", ok);
}

__device__ void add_entry(cs *T, int index, int i_, int j_, float value) {
  // printf("%d\n", index);
  T->i[index] = (csi)i_;
  T->p[index] = (csi)j_;
  T->x[index] = (double)value;
}

__global__ void gen_ABE_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor,
    float *V_WL_accessor, float *V_BL_accessor, int m, int n, float R_source,
    float R_line, cs *ABCD_matrix, double *E_matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  if (i < m && j < n) {
    int index = (i * n + j) * 5;
    // A matrix
    if (j == 0) {
      E_matrix[i * n] =
          (double)V_WL_accessor[i] / (double)R_source; // E matrix (partial)
      ABCD_matrix->i[index] = (csi)i * n;
      ABCD_matrix->p[index] = (csi)i * n;
      ABCD_matrix->x[index] = (double)100.0;
      // add_entry(ABCD_matrix, index, i * n, i * n,
      //           conductance_matrix_accessor[i][0] + 1.0 / R_source +
      //               1.0 / R_line);
    }
    // else {
    //   add_entry(ABCD_matrix, index, 0, 0, 0.0);
    // }
    // index++;
    // add_entry(ABCD_matrix, index, i * n + j, i * n + j,
    //           conductance_matrix_accessor[i][j] + 2.0 / R_line);
    // index++;
    // if (j < n - 1) {
    //   add_entry(ABCD_matrix, index, i * n + j + 1, i * n + j, -1.0 / R_line);
    //   index++;
    //   add_entry(ABCD_matrix, index, i * n + j, i * n + j + 1, -1.0 / R_line);
    // } else {
    //   add_entry(ABCD_matrix, index, i * n + j, i * n + j,
    //             conductance_matrix_accessor[i][j] + 1.0 / R_line);
    //   index++;
    //   add_entry(ABCD_matrix, index, 0, 0, 0.0);
    // }
    // index++;
    // // B matrix
    // add_entry(ABCD_matrix, index, i * n + j, i * n + j + (m * n),
    //           -conductance_matrix_accessor[i][j]);
  }
}

__global__ void gen_CDE_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor,
    float *V_WL_accessor, float *V_BL_accessor, int m, int n, float R_source,
    float R_line, cs *ABCD_matrix, double *E_matrix) {
  int j = threadIdx.x + blockIdx.x * blockDim.x; // for (int j = 0; j < n; j++)
  int i = threadIdx.y + blockIdx.y * blockDim.y; // for (int i = 0; i < m; i++)
  if (j < n && i < m) {
    int index = (5 * m * n) + ((j * m + i) * 4);
    // D matrix
    if (i == 0) {
      E_matrix[m * n + (j + 1) * m - 1] =
          (double)-V_BL_accessor[j] / (double)R_source; // E matrix (partial)
      add_entry(ABCD_matrix, index, m * n + (j * m), m * n + j,
                -1.0 / R_line - conductance_matrix_accessor[0][j]);
      index++;
      add_entry(ABCD_matrix, index, m * n + (j * m), m * n + j + n,
                1.0 / R_line);
      index++;
      add_entry(ABCD_matrix, index, 0, 0, 0.0);
    } else if (i < m - 1) {
      add_entry(ABCD_matrix, index, m * n + (j * m) + i,
                m * n + (n * (i - 1)) + j, 1.0 / R_line);
      index++;
      add_entry(ABCD_matrix, index, m * n + (j * m) + i,
                m * n + (n * (i + 1)) + j, 1.0 / R_line);
      index++;
      add_entry(ABCD_matrix, index, m * n + (j * m) + i, m * n + (n * i) + j,
                -conductance_matrix_accessor[i][j] - 2.0 / R_line);
    } else {
      add_entry(ABCD_matrix, index, m * n + (j * m) + m - 1,
                m * n + (n * (m - 2)) + j, 1 / R_line);
      index++;
      add_entry(ABCD_matrix, index, m * n + (j * m) + m - 1,
                m * n + (n * (m - 1)) + j,
                -1.0 / R_source - conductance_matrix_accessor[m - 1][j] -
                    1.0 / R_line);
      index++;
      add_entry(ABCD_matrix, index, 0, 0, 0.0);
    }
    index++;
    // C matrix
    add_entry(ABCD_matrix, index, j * m + i + (m * n), n * i + j,
              conductance_matrix_accessor[i][j]);
  }
}

__global__ void
construct_V_applied(torch::PackedTensorAccessor32<float, 2> V_applied_accessor,
                    double *V_accessor, int m, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  if (i < m && j < n) {
    V_applied_accessor[i][j] =
        (float)V_accessor[n * i + j] - (float)V_accessor[m * n + n * i + j];
  }
}

at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
                         at::Tensor V_BL, float R_source, float R_line) {
  assert(at::cuda::is_available());
  conductance_matrix = conductance_matrix.to(torch::Device("cuda:0"));
  V_WL = V_WL.to(torch::Device("cuda:0"));
  V_BL = V_BL.to(torch::Device("cuda:0"));
  int m = conductance_matrix.sizes()[0];
  int n = conductance_matrix.sizes()[1];
  torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor =
      conductance_matrix.packed_accessor32<float, 2>();
  float *V_WL_accessor = V_WL.data_ptr<float>();
  float *V_BL_accessor = V_BL.data_ptr<float>();
  int non_zero_elements =
      (5 * m * n) +
      (4 * m * n); // Uncompressed (with padding for CUDA execution).
  // When compressed, contains 8 * m * n - 2 * m - 2 * n unique values.
  std::cout << "here0" << std::endl;
  cs *ABCD_matrix;
  cudaMalloc(&ABCD_matrix, sizeof(cs));
  spalloc<<<1, 1>>>(ABCD_matrix, m, n, non_zero_elements);
  cudaSafeCall(cudaDeviceSynchronize());
  double *E_matrix;
  cudaMalloc(&E_matrix, sizeof(double) * 2 * m * n);
  cudaMemset(E_matrix, 0., sizeof(double) * 2 * m * n);
  std::cout << "here4" << std::endl;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads = prop.maxThreadsDim[0];
  dim3 grid;
  dim3 block;
  if (m * n > max_threads) {
    int n_grid = ceil_int_div(m * n, max_threads);
    grid = dim3(n_grid, n_grid, 1);
    block = dim3(ceil_int_div(m, n_grid), ceil_int_div(n, n_grid), 1);
  } else {
    grid = dim3(1, 1, 1);
    block = dim3(m, n, 1);
  }
  std::cout << "here" << std::endl;
  gen_ABE_kernel<<<grid, block>>>(conductance_matrix_accessor, V_WL_accessor,
                                  V_BL_accessor, m, n, R_source, R_line,
                                  ABCD_matrix, E_matrix);
  // gen_CDE_kernel<<<grid, block>>>(conductance_matrix_accessor, V_WL_accessor,
  //                                 V_BL_accessor, m, n, R_source, R_line,
  //                                 ABCD_matrix, E_matrix);
  cudaSafeCall(cudaDeviceSynchronize());
  auto c_ret = cudaGetLastError();
  if (c_ret) {
    std::cout << "Error: " << cudaGetErrorString(c_ret) << "-->";
  } else {
    std::cout << "Success." << std::endl;
  }
  double *values_host = (double *)malloc(sizeof(double));
  cudaMemcpy(values_host, &ABCD_matrix->x[0], sizeof(double),
             cudaMemcpyDeviceToHost);
  std::cout << values_host[0] << std::endl;
  // for (int i = 0; i < 10; i++) {
  //   std::cout << values_host[i] << std::endl;
  // }
  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 100);
  solve_sparse_linear_alt<<<1, 1>>>(ABCD_matrix, E_matrix);
  std::cout << "K" << std::endl;
  cudaSafeCall(cudaDeviceSynchronize());
  c_ret = cudaGetLastError();
  if (c_ret) {
    std::cout << "Error: " << cudaGetErrorString(c_ret) << "-->" << std::endl;
  } else {
    std::cout << "Success." << std::endl;
  }
  // double *E_matrix_host = (double *)malloc(sizeof(double) * 2 * m * n);
  // cudaMemcpy(E_matrix_host, E_matrix, sizeof(double) * 2 * m * n,
  //            cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 2 * m * n; i++) {
  //   std::cout << E_matrix_host[i] << std::endl;
  // }
  at::Tensor V_applied_tensor =
      at::zeros({m, n}, torch::TensorOptions().device(torch::kCUDA, 0));
  torch::PackedTensorAccessor32<float, 2> V_applied_accessor =
      V_applied_tensor.packed_accessor32<float, 2>();
  construct_V_applied<<<grid, block>>>(V_applied_accessor, E_matrix, m, n);
  cudaSafeCall(cudaDeviceSynchronize());
  c_ret = cudaGetLastError();
  if (c_ret) {
    std::cout << "Error: " << cudaGetErrorString(c_ret) << "-->";
  }
  // cudaSafeCall(cudaFree(ABCD_matrix));
  // cudaSafeCall(cudaFree(E_matrix));
  // cudaSafeCall(cudaFree(V_accessor));
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  return V_applied_tensor;
  // return at::sum(at::mul(V_applied_tensor, conductance_matrix), 0);
}