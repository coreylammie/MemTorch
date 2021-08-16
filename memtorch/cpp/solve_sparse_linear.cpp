#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>

#include <Eigen/SparseCore>

#include <Eigen/SparseCholesky>

void solve_sparse_linear() {
  std::vector<Eigen::Triplet<float>> triplet_list;
  triplet_list.reserve(1);
  triplet_list.push_back(Eigen::Triplet<float>(5, 5, 1.5f));

  Eigen::SparseMatrix<float> a(10, 10);
  a.setFromTriplets(triplet_list.begin(), triplet_list.end());
  Eigen::VectorXf b(10);
  b[0] = 2.0f;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
  solver.compute(a);
  Eigen::VectorXf x = solver.solve(b);
  std::cout << x << std::endl;
}

void solve_sparse_linear_bindings(py::module_ &m) {
  m.def("solve_sparse_linear", &solve_sparse_linear);
}