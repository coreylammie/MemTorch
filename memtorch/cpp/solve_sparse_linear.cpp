#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <iostream>

void solve_sparse_linear(Eigen::SparseMatrix<double> A, double *B_values,
                         int n) {
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> QR(
      A);
  QR.analyzePattern(A);
  QR.factorize(A);
  Eigen::Map<Eigen::VectorXd> B(B_values, n);
  Eigen::VectorXd X = QR.solve(B);
  memcpy(B_values, X.data(), sizeof(double) * n);
}

void solve_sparse_linear(Eigen::SparseMatrix<float> A, float *B_values, int n) {
  Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> QR(A);
  QR.analyzePattern(A);
  QR.factorize(A);
  Eigen::Map<Eigen::VectorXf> B(B_values, n);
  Eigen::VectorXf X = QR.solve(B);
  memcpy(B_values, X.data(), sizeof(float) * n);
}