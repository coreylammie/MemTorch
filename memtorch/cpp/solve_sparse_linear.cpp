#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>

#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

Eigen::VectorXf
solve_sparse_linear(Eigen::VectorXf A_indices_x, Eigen::VectorXf A_indices_y,
                    Eigen::VectorXf A_values, int A_nonzero_elements,
                    std::tuple<int, int> A_shape, Eigen::VectorXf B) {
  std::vector<Eigen::Triplet<float>> triplet_list;
  triplet_list.reserve(A_nonzero_elements);
#pragma omp parallel for
  for (int i = 0; i < A_nonzero_elements; i++) {
    triplet_list.push_back(
        Eigen::Triplet<float>(A_indices_x[i], A_indices_y[i], A_values[i]));
  }
  Eigen::SparseMatrix<float> A(std::get<0>(A_shape), std::get<1>(A_shape));
  A.setFromTriplets(triplet_list.begin(), triplet_list.end());
  Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
  solver.compute(A);
  return solver.solve(B);
}

void solve_sparse_linear_bindings(py::module_ &m) {
  m.def("solve_sparse_linear", [](at::Tensor A_indices_x,
                                  at::Tensor A_indices_y, at::Tensor A_values,
                                  std::tuple<int, int> A_shape, at::Tensor B) {
    int A_nonzero_elements = A_values.sizes()[0];
    Eigen::Map<Eigen::VectorXf> A_indices_x_vector(
        A_indices_x.data_ptr<float>(), A_nonzero_elements);
    Eigen::Map<Eigen::VectorXf> A_indices_y_vector(
        A_indices_y.data_ptr<float>(), A_nonzero_elements);
    Eigen::Map<Eigen::VectorXf> A_values_vector(A_values.data_ptr<float>(),
                                                A_nonzero_elements);
    Eigen::Map<Eigen::VectorXf> B_vector(B.data_ptr<float>(), B.sizes()[0]);
    Eigen::VectorXf X = solve_sparse_linear(
        A_indices_x_vector, A_indices_y_vector, A_values_vector,
        A_nonzero_elements, A_shape, B_vector);
    at::Tensor X_tensor = at::zeros({X.size()});
    memmove(X_tensor.data_ptr<float>(), X.data(), sizeof(float) * X.size());
    return X_tensor;
  });
}