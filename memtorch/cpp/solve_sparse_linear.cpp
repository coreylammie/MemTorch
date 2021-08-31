#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>

// #include <Eigen/SparseCore>

// #include <Eigen/SparseLU>

// #include "ST_TO_CC.h"

#include "CS/cs.h"

// #include "CS/cs_malloc.h"

typedef Eigen::Vector<int, Eigen::Dynamic> VectorXI;

// Eigen::VectorXf
// solve_sparse_linear(Eigen::VectorXf A_indices_x, Eigen::VectorXf A_indices_y,
//                     Eigen::VectorXf A_values, int A_nonzero_elements,
//                     std::tuple<int, int> A_shape, Eigen::VectorXf B) {
//   std::vector<Eigen::Triplet<float>> triplet_list;
//   triplet_list.reserve(A_nonzero_elements);
// #pragma omp parallel for
//   for (int i = 0; i < A_nonzero_elements; i++) {
//     triplet_list.push_back(
//         Eigen::Triplet<float>(A_indices_x[i], A_indices_y[i], A_values[i]));
//   }
//   Eigen::SparseMatrix<float> A(std::get<0>(A_shape), std::get<1>(A_shape));
//   A.setFromTriplets(triplet_list.begin(), triplet_list.end());
//   Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
//   solver.compute(A);
//   return solver.solve(B);
// }

// Eigen::VectorXf solve_sparse_linear(VectorXI A_indices_x, int m, int n,
//                                     Eigen::VectorXd B) {
//   return Eigen::VectorXf::Zero(m, n);
// }

Eigen::VectorXd solve_sparse_linear(int *A_indices_x_accessor,
                                    int *A_indices_y_accessor,
                                    double *A_values_accessor,
                                    int A_nonzero_elements, int m, int n,
                                    double *B_accessor) {
  // Construct a sparse matrix A
  cs *A = cs_spalloc(10, 10, 19, 1, 1);
  // std::cout << A->nz << std::endl;
  // std::cout << A->nzmax << std::endl;
  // std::cout << "---------------------" << std::endl;
  for (int i = 0; i < A_nonzero_elements; i++) {
    //   std::cout << A_values_accessor[i] << std::endl;
    cs_entry(A, A_indices_x_accessor[i], A_indices_y_accessor[i],
             A_values_accessor[i]);
  }
  // std::cout << "---------------------" << std::endl;
  cs *A_compressed = cs_compress(A);
  cs_spfree(A);
  for (int i = 0; i < A_nonzero_elements; i++) {
    std::cout << A->x[i] << std::endl;
  }
  std::cout << "---------------------" << std::endl;
  // Solve AX=B using QR factorization
  cs_qrsol(1, A_compressed, B_accessor);
  Eigen::Map<Eigen::VectorXd> X(B_accessor, n);
  return X;
}

void solve_sparse_linear_bindings(py::module_ &m) {
  m.def("solve_sparse_linear",
        [](at::Tensor A_indices_x, at::Tensor A_indices_y, at::Tensor A_values,
           std::tuple<int, int> A_shape, at::Tensor B) {
          // int A_nonzero_elements = A_values.sizes()[0];
          // A_indices_x = A_indices_x.to(torch::kInt32);
          // A_indices_y = A_indices_y.to(torch::kInt32);
          int A_nonzero_elements = 19;
          int A_indices_x_accessor[] = {1, 6, 5, 4, 9, 0, 2, 6, 8, 1,
                                        2, 7, 4, 2, 3, 5, 7, 6, 7};
          int A_indices_y_accessor[] = {0, 0, 1, 2, 2, 3, 4, 5, 5, 6,
                                        6, 6, 7, 8, 8, 8, 8, 9, 9};
          double A_values_accessor[] = {
              -0.563833742338016, -0.591908940985609, 0.559849215154234,
              -0.253963929329736, -0.746974224826767, 0.148071275980455,
              -0.730804912317641, 0.711443743939293,  -1.04374045545080,
              -0.965352009768567, -0.565568775086541, 0.860497537960044,
              1.03140136113829,   -0.941106875100149, -0.164924307450810,
              0.205878365083051,  0.233891294833039,  -1.57903145117562,
              0.385992304174899};
          int m = 10;
          int n = 10;
          Eigen::VectorXd X = solve_sparse_linear(
              A_indices_x_accessor, A_indices_y_accessor, A_values_accessor,
              A_nonzero_elements, m, n, B.data_ptr<double>());
          std::cout << X << std::endl;

          // int m = std::get<0>(A_shape);
          // int n = std::get<1>(A_shape);
          // Eigen::VectorXf X = solve_sparse_linear(
          //     A_indices_x.data_ptr<int>(), A_indices_y.data_ptr<int>(),
          //     A_values.data_ptr<double>(), A_nonzero_elements, m, n,
          //     B.data_ptr<double>());
          at::Tensor X_tensor = at::zeros({X.size()});
          // memmove(X_tensor.data_ptr<float>(), X.data(), sizeof(double) *
          // X.size());
          return X_tensor;
        });
}