#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

void solve_sparse_linear(Eigen::SparseMatrix<double> A, double *B_values, int n);
void solve_sparse_linear(Eigen::SparseMatrix<float> A, float *B_values, int n);