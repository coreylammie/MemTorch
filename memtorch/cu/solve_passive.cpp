#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>

#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

#include "solve_passive_kernels.cuh"

Eigen::VectorXf solve_sparse_linear(Eigen::SparseMatrix<float> A,
                                    Eigen::VectorXf B) {
  Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
  solver.compute(A);
  return solver.solve(B);
}

void solve_passive_bindings(
    py::module_ &m) { // Temporary bindings for debugging purposes
  m.def(
      "solve_passive",
      [&](at::Tensor conductance_matrix, at::Tensor V_WL, at::Tensor V_BL,
          float R_source, float R_line) {
        return solve_passive(conductance_matrix, V_WL, V_BL, R_source, R_line);
      },
      py::arg("conductance_matrix"), py::arg("V_WL"), py::arg("V_BL"),
      py::arg("R_source"), py::arg("R_line"));
}