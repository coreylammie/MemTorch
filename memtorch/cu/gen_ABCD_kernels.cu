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

#define cudaSafeCall(call)                                                     \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__             \
                << "): " << cudaGetErrorString(err);                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int ceil_int_div(int a, int b);

class Triplet {
public:
  __host__ __device__ Triplet() : m_row(0), m_col(0), m_value(0) {}

  __host__ __device__ Triplet(int i, int j, float v)
      : m_row(i), m_col(j), m_value(v) {}

  __host__ __device__ const int &row() { return m_row; }
  __host__ __device__ const int &col() { return m_col; }
  __host__ __device__ const float &value() { return m_value; }
  int m_row, m_col;
  float m_value;
  // protected:
  //   int m_row, m_col;
  //   float m_value;
};

typedef Triplet sparse_element;

// __global__ void gen_AB_kernel(
//     torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor, int
//     m, int n, float R_source, float R_line, sparse_element *ABCD_matrix) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m;
//   i++) int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j <
//   n; j++) printf("%d, %d\n", i, j);

// }

__global__ void gen_AB_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor, int m,
    int n, float R_source, float R_line, sparse_element *ABCD_matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  // int current_index;
  if (i < m && j < n) {
    int index = (i * n + j) * 5;
    // A matrix
    if (j == 0) {
      ABCD_matrix[index] = sparse_element(i * n, i * n,
                                          conductance_matrix_accessor[i][0] +
                                              1.0f / R_source + 1.0f / R_line);
    } else {
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    ABCD_matrix[index] =
        sparse_element(i * n + j, i * n + j,
                       conductance_matrix_accessor[i][j] + 2.0f / R_line);
    index++;
    if (j < n - 1) {
      ABCD_matrix[index] =
          sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line);
      index++;
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line);
    } else {
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j,
                         conductance_matrix_accessor[i][j] + 1.0 / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // B matrix
    ABCD_matrix[index] = sparse_element(i * n + j, i * n + j + (m * n),
                                        -conductance_matrix_accessor[i][j]);
  }
}

// __global__ void gen_CD_kernel(
//     torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor, int
//     m, int n, float R_source, float R_line, int *index, sparse_element
//     *ABCD_matrix) {
//   int j = threadIdx.x + blockIdx.x * blockDim.x; // for (int j = 0; j < n;
//   j++) int i = threadIdx.y + blockIdx.y * blockDim.y; // for (int i = 0; i <
//   m; i++)
//   // D matrix
//   if (i == 0) {
//     ABCD_matrix[&index] =
//         sparse_element(m * n + (j * m), m * n + j,
//                        -1.0f / R_line - conductance_matrix_accessor[0][j]);
//     &index++;
//     ABCD_matrix[&index] =
//         sparse_element(m * n + (j * m), m * n + j + n, 1.0f / R_line);
//     &index++;
//     ABCD_matrix[&index] = sparse_element(m * n + (j * m) + m - 1,
//                                          m * n + (n * (m - 2)) + j, 1 /
//                                          R_line);
//     &index++;
//     ABCD_matrix[&index] = sparse_element(
//         m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
//         -1.0f / R_source - conductance_matrix_accessor[m - 1][j] -
//             1.0f / R_line);
//     &index++;
//   } else if (i < m - 1) {
//     ABCD_matrix[&index] = sparse_element(
//         m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 1.0f / R_line);
//     &index++;
//     ABCD_matrix[&index] = sparse_element(
//         m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 1.0f / R_line);
//     ABCD_matrix[&index] =
//         sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
//                        -conductance_matrix_accessor[i][j] - 2.0f / R_line);
//     &index++;
//   }
//   // C matrix
//   ABCD_matrix[&index] = sparse_element(j * m + i + (m * n), n * i + j,
//                                        conductance_matrix_accessor[i][j]);
//   &index++;
// }

at::Tensor gen_ABCD(at::Tensor conductance_matrix, int m, int n, float R_source,
                    float R_line) {
  conductance_matrix = conductance_matrix.to(torch::Device("cuda:0"));
  torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor =
      conductance_matrix.packed_accessor32<float, 2>();
  int non_zero_elements = 5 * m * n; // With padding for CUDA execution, 8 * m *
                                     // n - 2 * m - 2 * n unique values.
  sparse_element *ABCD_matrix;
  cudaMalloc(&ABCD_matrix, sizeof(sparse_element) * non_zero_elements);
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
  gen_AB_kernel<<<grid, block>>>(conductance_matrix_accessor, m, n, R_source,
                                 R_line, ABCD_matrix);
  cudaSafeCall(cudaDeviceSynchronize());
  cudaError_t c_ret = cudaGetLastError();
  if (c_ret) {
    std::cout << "Error: " << cudaGetErrorString(c_ret) << "-->";
  }
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

  // for (int i = 0; i < non_zero_elements; i++) {
  //   std::cout << ABCD_matrix[i].row() << " " << ABCD_matrix[i].col() << " "
  //             << ABCD_matrix[i].value() << std::endl;
  // }

  return conductance_matrix;
}

void gen_ABCD_bindings(
    py::module_ &m) { // Temporary bindings for debugging purposes
  m.def(
      "gen_ABCD",
      [&](at::Tensor conductance_matrix, int m, int n, float R_source,
          float R_line) {
        return gen_ABCD(conductance_matrix, m, n, R_source, R_line);
        // Eigen::SparseMatrix<float> ABCD(2 * m * n, 2 * m * n);
        // ABCD.setFromTriplets(&ABCD_matrix[0],
        //                      &ABCD_matrix[non_zero_elements + 1]);
      },
      py::arg("conductance_matrix"), py::arg("m"), py::arg("n"),
      py::arg("R_source"), py::arg("R_line"));
}