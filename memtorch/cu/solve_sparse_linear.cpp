#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <iostream>


void solve_sparse_linear(Eigen::SparseMatrix<double> A, double *B_values,
                         int n) {
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> LU(A);
  LU.analyzePattern(A);
  LU.factorize(A);
  Eigen::Map<Eigen::VectorXd> B(B_values, n);
  Eigen::VectorXd X = LU.solve(B);
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    B_values[i] = X[i];
  }
}

void solve_sparse_linear(Eigen::SparseMatrix<float> A, float *B_values, int n) {
  Eigen::SparseLU<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> LU(A);
  LU.analyzePattern(A);
  LU.factorize(A);
  Eigen::Map<Eigen::VectorXf> B(B_values, n);
  Eigen::VectorXf X = LU.solve(B);
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    B_values[i] = X[i];
  }
}