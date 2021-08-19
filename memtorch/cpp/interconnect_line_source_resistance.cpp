#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

typedef Eigen::Triplet<float> sparse_element;

at::Tensor gen_ABCD_E(at::Tensor conductance_matrix, at::Tensor V_WL,
                      at::Tensor V_BL, float R_source, float R_line,
                      bool det_readout_currents) {
  int m = conductance_matrix.sizes()[0];
  int n = conductance_matrix.sizes()[1];
  Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor, Eigen::Stride<1, Eigen::Dynamic>>
      conductance_matrix_accessor(conductance_matrix.data_ptr<float>(), m, n,
                                  Eigen::Stride<1, Eigen::Dynamic>(1, n));
  float *V_WL_accessor = V_WL.data_ptr<float>();
  float *V_BL_accessor = V_BL.data_ptr<float>();
  int non_zero_elements = 8 * m * n - 2 * m - 2 * n;
  std::vector<Eigen::Triplet<float>> ABCD_matrix;
  ABCD_matrix.reserve(non_zero_elements);
  Eigen::VectorXf E_matrix = Eigen::VectorXf::Zero(2 * m * n);
  // A, B, and E (partial) matrices
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    // A matrix
    ABCD_matrix.push_back(sparse_element(i * n, i * n,
                                         conductance_matrix_accessor(i, 0) +
                                             1.0f / R_source + 1.0f / R_line));
    ABCD_matrix.push_back(sparse_element(i * n + 1, i * n, -1.0f / R_line));
    ABCD_matrix.push_back(sparse_element(i * n, i * n + 1, -1.0f / R_line));
    ABCD_matrix.push_back(
        sparse_element(i * n + (n - 1), i * n + (n - 1),
                       conductance_matrix_accessor(i, n - 1) + 1.0 / R_line));
    // B matrix
    ABCD_matrix.push_back(sparse_element(i * n, i * n + (m * n),
                                         -conductance_matrix_accessor(i, 0)));
    ABCD_matrix.push_back(
        sparse_element(i * n + (n - 1), i * n + (n - 1) + (m * n),
                       -conductance_matrix_accessor(i, n - 1)));
    // E matrix
    E_matrix(i * n) = V_WL_accessor[i] / R_source;
#pragma omp for nowait
    for (int j = 1; j < n - 1; j++) {
      // A matrix
      ABCD_matrix.push_back(
          sparse_element(i * n + j, i * n + j,
                         conductance_matrix_accessor(i, j) + 2.0f / R_line));
      ABCD_matrix.push_back(
          sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line));
      ABCD_matrix.push_back(
          sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line));
      // B matrix
      ABCD_matrix.push_back(sparse_element(i * n + j, i * n + j + (m * n),
                                           -conductance_matrix_accessor(i, j)));
    }
  }
  // C, D, and E (partial) matrices
#pragma omp parallel for
  for (int j = 0; j < n; j++) {
    // D matrix
    ABCD_matrix.push_back(
        sparse_element(m * n + (j * m), m * n + j,
                       -1.0f / R_line - conductance_matrix_accessor(0, j)));
    ABCD_matrix.push_back(
        sparse_element(m * n + (j * m), m * n + j + n, 1.0f / R_line));
    ABCD_matrix.push_back(sparse_element(
        m * n + (j * m) + m - 1, m * n + (n * (m - 2)) + j, 1 / R_line));
    ABCD_matrix.push_back(sparse_element(
        m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
        -1.0f / R_source - conductance_matrix_accessor(m - 1, j) -
            1.0f / R_line));
    // C matrix
    ABCD_matrix.push_back(
        sparse_element(j * m + (m * n), j, conductance_matrix_accessor(0, j)));
    ABCD_matrix.push_back(
        sparse_element(j * m + (m - 1) + (m * n), n * (m - 1) + j,
                       conductance_matrix_accessor(m - 1, j)));
    // E matrix
    E_matrix(m * n + (j + 1) * m - 1) = -V_BL_accessor[j] / R_source;
#pragma omp for nowait
    for (int i = 1; i < m - 1; i++) {
      // D matrix
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 1.0f / R_line));
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 1.0f / R_line));
      ABCD_matrix.push_back(
          sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
                         -conductance_matrix_accessor(i, j) - 2.0f / R_line));
      // C matrix
      ABCD_matrix.push_back(sparse_element(j * m + i + (m * n), n * i + j,
                                           conductance_matrix_accessor(i, j)));
    }
  }
  // Solve (ABCD)V = E
  Eigen::SparseMatrix<float> ABCD(2 * m * n, 2 * m * n);
  ABCD.setFromTriplets(ABCD_matrix.begin(), ABCD_matrix.end());
  Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
  solver.compute(ABCD);
  Eigen::VectorXf V = solver.solve(E_matrix);
  at::Tensor V_applied_tensor = at::zeros({m, n});
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
#pragma omp for nowait
    for (int j = 0; j < n; j++) {
      V_applied_tensor.index_put_({i, j}, V[n * i + j] - V[m * n + n * i + j]);
    }
  }
  // if (det_readout_currents) {
  return V_applied_tensor;
  // }
}

void interconnect_line_source_resistance_bindings(py::module_ &m) {
  m.def("gen_ABCD_E", &gen_ABCD_E);
}