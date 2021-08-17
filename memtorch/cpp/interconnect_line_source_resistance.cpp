#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

typedef Eigen::Triplet<float> sparse_element;

at::Tensor gen_ABCD_E(at::Tensor conductance_matrix, at::Tensor V_WL,
                      at::Tensor V_BL, float R_source, float R_line) {
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
  Eigen::VectorXd mn_range = Eigen::VectorXd::LinSpaced(0, m * n - 1, m * n);
  Eigen::VectorXd m_range = Eigen::VectorXd::LinSpaced(0, m - 1, m);
  Eigen::VectorXd n_range = Eigen::VectorXd::LinSpaced(0, n - 1, n);
// A matrix
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    ABCD_matrix.push_back(sparse_element(i * n, i * n,
                                         conductance_matrix_accessor(i, 0) +
                                             1.0f / R_source + 1.0f / R_line));
    ABCD_matrix.push_back(sparse_element(i * n + 1, i * n, -1.0f / R_line));
    ABCD_matrix.push_back(sparse_element(i * n, i * n + 1, -1.0f / R_line));
    ABCD_matrix.push_back(
        sparse_element(i * n + (n - 1), i * n + (n - 1),
                       conductance_matrix_accessor(i, n - 1) + 1.0 / R_line));
#pragma omp for nowait
    for (int j = 1; j < n - 1; j++) {
      ABCD_matrix.push_back(
          sparse_element(i * n + j, i * n + j,
                         conductance_matrix_accessor(i, j) + 2.0f / R_line));
      ABCD_matrix.push_back(
          sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line));
      ABCD_matrix.push_back(
          sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line));
    }
  }
  // B and C matrices
  int j;
  int k;
#pragma omp parallel for
  for (int i = 0; i < m * n; i++) {
    j = i % m;
    k = i / m;
    ABCD_matrix.push_back(sparse_element(
        i, i + (m * n), -conductance_matrix_accessor(j, k))); // B matrix
    ABCD_matrix.push_back(
        sparse_element(i + (m * n), n * (j - 1) + k,
                       conductance_matrix_accessor(j, k))); // C matrix
  }
// D matrix
#pragma omp parallel for
  for (int j = 0; j < n; j++) {
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
#pragma omp for nowait
    for (int i = 1; i < m - 1; i++) {
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 1.0f / R_line));
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 1.0f / R_line));
      ABCD_matrix.push_back(
          sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
                         -conductance_matrix_accessor(i, j) - 2.0f / R_line));
    }
  }
  // E matrix
  Eigen::VectorXf E_matrix = Eigen::VectorXf::Zero(2 * m * n);
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    E_matrix(i * n) = V_WL_accessor[i] / R_source;
  }
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    E_matrix(m * n + (i + 1) * m - 1) = -V_BL_accessor[i] / R_source;
  }
  // Solve (ABCD)V = E
  Eigen::SparseMatrix<float> ABCD(2 * m * n, 2 * m * n);
  ABCD.setFromTriplets(ABCD_matrix.begin(), ABCD_matrix.end());
  Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
  solver.compute(ABCD);
  Eigen::VectorXf V = solver.solve(E_matrix);
  at::Tensor V_tensor = at::zeros({V.size()});
  memcpy(V_tensor.data_ptr<float>(), V.data(), sizeof(float) * V.size());
  return V_tensor;
}

void interconnect_line_source_resistance_bindings(py::module_ &m) {
  m.def("gen_ABCD_E", &gen_ABCD_E);
}