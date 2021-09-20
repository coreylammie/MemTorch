#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>

#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <ST_TO_CC.cuh>

typedef Eigen::Vector<int, Eigen::Dynamic> VectorXI;

Eigen::VectorXd solve_sparse_linear_(int *A_indices_x_accessor,
                                     int *A_indices_y_accessor,
                                     double *A_values_accessor,
                                     int A_nonzero_elements, int m, int n,
                                     double *B_accessor) {
  // Construct a sparse matrix A
  Eigen::Map<Eigen::SparseMatrix<double>> A(
      m, n, A_nonzero_elements, A_indices_x_accessor, A_indices_y_accessor,
      A_values_accessor);
  Eigen::Map<Eigen::VectorXd> B(B_accessor, n);
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QR(
      A);
  QR.analyzePattern(A);
  QR.factorize(A);
  Eigen::VectorXd X = QR.solve(B);
  return X;
}

void solve_sparse_linear_bindings(py::module_ &m) {
  m.def("solve_sparse_linear", [](at::Tensor A_indices_x,
                                  at::Tensor A_indices_y, at::Tensor A_values,
                                  std::tuple<int, int> A_shape, at::Tensor B) {
    int A_nonzero_elements = 19;
    int A_indices_x_accessor[] = {1, 6, 5, 4, 9, 0, 2, 6, 8, 1,
                                  2, 7, 4, 2, 3, 5, 7, 6, 7, 0};
    int A_indices_y_accessor[] = {0, 0, 1, 2, 2, 3, 4, 5, 5, 6,
                                  6, 6, 7, 8, 8, 8, 8, 9, 9, 0};
    double A_values_accessor[] = {-0.563833742338016, -0.591908940985609,
                                  0.559849215154234,  -0.253963929329736,
                                  -0.746974224826767, 0.148071275980455,
                                  -0.730804912317641, 0.711443743939293,
                                  -1.04374045545080,  -0.965352009768567,
                                  -0.565568775086541, 0.860497537960044,
                                  1.03140136113829,   -0.941106875100149,
                                  -0.164924307450810, 0.205878365083051,
                                  0.233891294833039,  -1.57903145117562,
                                  0.385992304174899,  10.0};
    int m = 10;
    int n = 10;
    int ncc = st_to_cc_size(A_nonzero_elements, A_indices_x_accessor,
                            A_indices_y_accessor);
    int *icc = (int *)malloc(ncc * sizeof(int));
    int *jcc = (int *)malloc(ncc * sizeof(int));
    int *ccc = (int *)malloc((n + 1) * sizeof(int));
    double *acc = (double *)malloc(ncc * sizeof(double));
    st_to_cc_index(A_nonzero_elements, A_indices_x_accessor,
                   A_indices_y_accessor, ncc, n, icc, ccc, jcc);
    free(jcc);
    st_to_cc_values(A_nonzero_elements, A_indices_x_accessor,
                    A_indices_y_accessor, A_values_accessor, ncc, n, icc, ccc,
                    acc);
    free(acc);
    Eigen::VectorXd X = solve_sparse_linear_(ccc, icc, A_values_accessor, ncc,
                                             m, n, B.data_ptr<double>());
    // free(icc);
    // free(jcc);
    at::Tensor X_tensor = at::zeros({n});
    for (int i = 0; i < n; i++) {
      X_tensor[i] = (float)X[i];
    }
    return X_tensor;
  });
}