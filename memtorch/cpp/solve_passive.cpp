#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

typedef Eigen::Triplet<float> sparse_element;
typedef std::vector<Eigen::Triplet<float>> triplet_vector;

triplet_vector gen_ABCDE(Eigen::MatrixXf conductance_matrix_accessor, int m,
                         int n, float *V_WL_accessor, float *V_BL_accessor,
                         float R_source, float R_line,
                         triplet_vector ABCD_matrix,
                         float *E_matrix_accessor = NULL) {
// A, B, and E (partial) matrices
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    // A matrix
    if (R_source == 0) {
      ABCD_matrix.push_back(sparse_element(
          i * n, i * n, conductance_matrix_accessor(i, 0) + 1.0f / R_line));
    } else if (R_line == 0) {
      ABCD_matrix.push_back(sparse_element(
          i * n, i * n, conductance_matrix_accessor(i, 0) + 1.0f / R_source));
    } else {
      ABCD_matrix.push_back(sparse_element(
          i * n, i * n,
          conductance_matrix_accessor(i, 0) + 1.0f / R_source + 1.0f / R_line));
    }
    if (R_line == 0) {
      ABCD_matrix.push_back(sparse_element(i * n + 1, i * n, 0));
      ABCD_matrix.push_back(sparse_element(i * n, i * n + 1, 0));
      ABCD_matrix.push_back(
          sparse_element(i * n + (n - 1), i * n + (n - 1),
                         conductance_matrix_accessor(i, n - 1)));
    } else {
      ABCD_matrix.push_back(sparse_element(i * n + 1, i * n, -1.0f / R_line));
      ABCD_matrix.push_back(sparse_element(i * n, i * n + 1, -1.0f / R_line));
      ABCD_matrix.push_back(
          sparse_element(i * n + (n - 1), i * n + (n - 1),
                         conductance_matrix_accessor(i, n - 1) + 1.0 / R_line));
    }
    // B matrix
    ABCD_matrix.push_back(sparse_element(i * n, i * n + (m * n),
                                         -conductance_matrix_accessor(i, 0)));
    ABCD_matrix.push_back(
        sparse_element(i * n + (n - 1), i * n + (n - 1) + (m * n),
                       -conductance_matrix_accessor(i, n - 1)));
    // E matrix
    if (E_matrix_accessor != NULL) {
      if (R_source == 0) {
        E_matrix_accessor[i * n] = V_WL_accessor[i];
      } else {
        E_matrix_accessor[i * n] = V_WL_accessor[i] / R_source;
      }
    }
#pragma omp for nowait
    for (int j = 1; j < n - 1; j++) {
      // A matrix
      if (R_line == 0) {
        ABCD_matrix.push_back(sparse_element(
            i * n + j, i * n + j, conductance_matrix_accessor(i, j)));
        ABCD_matrix.push_back(sparse_element(i * n + j + 1, i * n + j, 0));
        ABCD_matrix.push_back(sparse_element(i * n + j, i * n + j + 1, 0));
      } else {
        ABCD_matrix.push_back(
            sparse_element(i * n + j, i * n + j,
                           conductance_matrix_accessor(i, j) + 2.0f / R_line));
        ABCD_matrix.push_back(
            sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line));
        ABCD_matrix.push_back(
            sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line));
      }
      // B matrix
      ABCD_matrix.push_back(sparse_element(i * n + j, i * n + j + (m * n),
                                           -conductance_matrix_accessor(i, j)));
    }
  }
  // C, D, and E (partial) matrices
#pragma omp parallel for
  for (int j = 0; j < n; j++) {
    // D matrix
    if (R_line == 0) {
      ABCD_matrix.push_back(sparse_element(m * n + (j * m), m * n + j,
                                           -conductance_matrix_accessor(0, j)));
      ABCD_matrix.push_back(sparse_element(m * n + (j * m), m * n + j + n, 0));
      ABCD_matrix.push_back(sparse_element(m * n + (j * m) + m - 1,
                                           m * n + (n * (m - 2)) + j, 0));
    } else {
      ABCD_matrix.push_back(
          sparse_element(m * n + (j * m), m * n + j,
                         -1.0f / R_line - conductance_matrix_accessor(0, j)));
      ABCD_matrix.push_back(
          sparse_element(m * n + (j * m), m * n + j + n, 1.0f / R_line));
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 2)) + j, 1.0f / R_line));
    }
    if (R_source == 0) {
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
          -conductance_matrix_accessor(m - 1, j) - 1.0f / R_line));
    } else if (R_line == 0) {
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
          -1.0f / R_source - conductance_matrix_accessor(m - 1, j)));
    } else {
      ABCD_matrix.push_back(sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
          -1.0f / R_source - conductance_matrix_accessor(m - 1, j) -
              1.0f / R_line));
    }
    // C matrix
    ABCD_matrix.push_back(
        sparse_element(j * m + (m * n), j, conductance_matrix_accessor(0, j)));
    ABCD_matrix.push_back(
        sparse_element(j * m + (m - 1) + (m * n), n * (m - 1) + j,
                       conductance_matrix_accessor(m - 1, j)));
    // E matrix
    if (E_matrix_accessor != NULL) {
      if (R_source == 0) {
        E_matrix_accessor[m * n + (j + 1) * m - 1] = -V_BL_accessor[j];
      } else {
        E_matrix_accessor[m * n + (j + 1) * m - 1] =
            -V_BL_accessor[j] / R_source;
      }
    }
#pragma omp for nowait
    for (int i = 1; i < m - 1; i++) {
      // D matrix
      if (R_line == 0) {
        ABCD_matrix.push_back(
            sparse_element(m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 0));
        ABCD_matrix.push_back(
            sparse_element(m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 0));
        ABCD_matrix.push_back(
            sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
                           -conductance_matrix_accessor(i, j)));
      } else {
        ABCD_matrix.push_back(sparse_element(
            m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 1.0f / R_line));
        ABCD_matrix.push_back(sparse_element(
            m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 1.0f / R_line));
        ABCD_matrix.push_back(
            sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
                           -conductance_matrix_accessor(i, j) - 2.0f / R_line));
      }
      // C matrix
      ABCD_matrix.push_back(sparse_element(j * m + i + (m * n), n * i + j,
                                           conductance_matrix_accessor(i, j)));
    }
  }
  return ABCD_matrix;
}

at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
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
  triplet_vector ABCD_matrix;
  ABCD_matrix.reserve(non_zero_elements);
  float *E_matrix_accessor = (float *)malloc(sizeof(float) * (2 * m * n));
#pragma omp parallel for
  for (int i = 0; i < (2 * m * n); i++) {
    E_matrix_accessor[i] = 0;
  }
  ABCD_matrix =
      gen_ABCDE(conductance_matrix_accessor, m, n, V_WL_accessor, V_BL_accessor,
                R_source, R_line, ABCD_matrix, E_matrix_accessor);
  Eigen::Map<Eigen::VectorXf> E_matrix(E_matrix_accessor, (2 * m * n));
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
  if (!det_readout_currents) {
    return V_applied_tensor;
  } else {
    return at::sum(at::mul(V_applied_tensor, conductance_matrix), 0);
  }
}

at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
                         at::Tensor V_BL, float R_source, float R_line,
                         int n_input_batches) {
  int m = conductance_matrix.sizes()[0];
  int n = conductance_matrix.sizes()[1];
  Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor, Eigen::Stride<1, Eigen::Dynamic>>
      conductance_matrix_accessor(conductance_matrix.data_ptr<float>(), m, n,
                                  Eigen::Stride<1, Eigen::Dynamic>(1, n));
  Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor, Eigen::Stride<1, Eigen::Dynamic>>
      V_WL_accessor(V_WL.data_ptr<float>(), n_input_batches, m,
                    Eigen::Stride<1, Eigen::Dynamic>(1, m));
  Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor, Eigen::Stride<1, Eigen::Dynamic>>
      V_BL_accessor(V_BL.data_ptr<float>(), n_input_batches, n,
                    Eigen::Stride<1, Eigen::Dynamic>(1, n));
  int non_zero_elements = 8 * m * n - 2 * m - 2 * n;
  triplet_vector ABCD_matrix;
  ABCD_matrix.reserve(non_zero_elements);
  ABCD_matrix = gen_ABCDE(conductance_matrix_accessor, m, n, NULL, NULL,
                          R_source, R_line, ABCD_matrix, NULL);
  // Solve (ABCD)V = E
  Eigen::SparseMatrix<float> ABCD(2 * m * n, 2 * m * n);
  ABCD.setFromTriplets(ABCD_matrix.begin(), ABCD_matrix.end());
  Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
  solver.compute(ABCD);
  at::Tensor out = at::zeros({n_input_batches, n});
#pragma omp parallel for
  for (int i = 0; i < n_input_batches; i++) {
    Eigen::VectorXf E_matrix = Eigen::VectorXf::Zero(2 * m * n);
    for (int j = 0; j < m; j++) {
      E_matrix(j * n) = V_WL_accessor(i, j) / R_source;
    }
    for (int k = 0; k < n; k++) {
      E_matrix(m * n + (k + 1) * m - 1) = -V_BL_accessor(i, k) / R_source;
    }
    Eigen::VectorXf V = solver.solve(E_matrix);
    at::Tensor V_applied_tensor = at::zeros({m, n});
#pragma omp parallel for
    for (int j = 0; j < m; j++) {
#pragma omp for nowait
      for (int k = 0; k < n; k++) {
        V_applied_tensor.index_put_({j, k}, V_applied_tensor.index({j, k}) +
                                                V[n * j + k] -
                                                V[m * n + n * j + k]);
      }
    }
    out.index_put_({i, torch::indexing::Slice()},
                   at::sum(at::mul(V_applied_tensor, conductance_matrix), 0));
  }
  return out;
}

void solve_passive_bindings(py::module_ &m) {
  m.def(
      "solve_passive",
      [&](at::Tensor conductance_matrix, at::Tensor V_WL, at::Tensor V_BL,
          float R_source, float R_line, bool det_readout_currents) {
        return solve_passive(conductance_matrix, V_WL, V_BL, R_source, R_line,
                             det_readout_currents);
      },
      py::arg("conductance_matrix"), py::arg("V_WL"), py::arg("V_BL"),
      py::arg("R_source"), py::arg("R_line"),
      py::arg("det_readout_currents") = true);
  m.def(
      "solve_passive",
      [&](at::Tensor conductance_matrix, at::Tensor V_WL, at::Tensor V_BL,
          float R_source, float R_line, int n_input_batches) {
        return solve_passive(conductance_matrix, V_WL, V_BL, R_source, R_line,
                             n_input_batches);
      },
      py::arg("conductance_matrix"), py::arg("V_WL"), py::arg("V_BL"),
      py::arg("R_source"), py::arg("R_line"), py::arg("n_input_batches"));
}