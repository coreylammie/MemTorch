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

#include <cs.h>

__global__ void solve_sparse_linear_alt(csi *ABCD_matrix_i_accessor,
                                        csi *ABCD_matrix_j_accessor,
                                        double *ABCD_matrix_value_accessor,
                                        double *E_matrix, int m, int n,
                                        int non_zero_elements) {
  cs *ABCD_matrix = (cs *)malloc(sizeof(cs));
  ABCD_matrix->m = (csi)(m);
  ABCD_matrix->n = (csi)(n);
  ABCD_matrix->nzmax = (csi)non_zero_elements;
  ABCD_matrix->nz = (csi)non_zero_elements;
  swap(ABCD_matrix->i, ABCD_matrix_i_accessor);
  swap(ABCD_matrix->p, ABCD_matrix_j_accessor);
  swap(ABCD_matrix->x, ABCD_matrix_value_accessor);
  cs *ABCD_matrix_compressed = cs_compress(ABCD_matrix);
  cs_spfree(ABCD_matrix);
  cs_qrsol(1, ABCD_matrix_compressed, E_matrix);
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
  int A_nonzero_elements = 20;
  csi A_indices_x_accessor_host[] = {1, 1, 6, 5, 4, 9, 0, 2, 6, 8,
                                     1, 2, 7, 4, 2, 3, 5, 7, 6, 7};
  csi A_indices_y_accessor_host[] = {0, 0, 0, 1, 2, 2, 3, 4, 5, 5,
                                     6, 6, 6, 7, 8, 8, 8, 8, 9, 9};
  double A_values_accessor_host[] = {0,
                                     -0.563833742338016,
                                     -0.591908940985609,
                                     0.559849215154234,
                                     -0.253963929329736,
                                     -0.746974224826767,
                                     0.148071275980455,
                                     -0.730804912317641,
                                     0.711443743939293,
                                     -1.04374045545080,
                                     -0.965352009768567,
                                     -0.565568775086541,
                                     0.860497537960044,
                                     1.03140136113829,
                                     -0.941106875100149,
                                     -0.164924307450810,
                                     0.205878365083051,
                                     0.233891294833039,
                                     -1.57903145117562,
                                     0.385992304174899};
  csi *A_indices_x_accessor;
  cudaMalloc(&A_indices_x_accessor, sizeof(csi) * A_nonzero_elements);
  cudaMemcpy(A_indices_x_accessor, A_indices_x_accessor_host,
             sizeof(csi) * A_nonzero_elements, cudaMemcpyHostToDevice);
  csi *A_indices_y_accessor;
  cudaMalloc(&A_indices_y_accessor, sizeof(csi) * A_nonzero_elements);
  cudaMemcpy(A_indices_y_accessor, A_indices_y_accessor_host,
             sizeof(csi) * A_nonzero_elements, cudaMemcpyHostToDevice);
  double *A_values_accessor;
  cudaMalloc(&A_values_accessor, sizeof(double) * A_nonzero_elements);
  cudaMemcpy(A_values_accessor, A_values_accessor_host,
             sizeof(double) * A_nonzero_elements, cudaMemcpyHostToDevice);
  int m = 10;
  int n = 10;
  double *E_matrix_host = (double *)malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) {
    E_matrix_host[i] = 1.0;
  }
  double *E_matrix;
  cudaMalloc(&E_matrix, n * sizeof(double));
  cudaMemcpy(E_matrix, E_matrix_host, n * sizeof(double),
             cudaMemcpyHostToDevice);
  solve_sparse_linear_alt<<<1, 1>>>(A_indices_x_accessor, A_indices_y_accessor,
                                    A_values_accessor, E_matrix, m, n,
                                    A_nonzero_elements);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           cudaGetErrorString(cudaerr));
  cudaMemcpy(E_matrix_host, E_matrix, sizeof(double) * n,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    std::cout << E_matrix_host[i] << std::endl;
  }
  at::Tensor V_applied_tensor =
      at::zeros({m, n}, torch::TensorOptions().device(torch::kCUDA, 0));
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  return V_applied_tensor;
}