#include <Eigen/Core>

void solve_sparse_linear_bindings(py::module_ &m);
Eigen::VectorXf
solve_sparse_linear(Eigen::VectorXf A_indices_x, Eigen::VectorXf A_indices_y,
                    Eigen::VectorXf A_values, int A_nonzero_elements,
                    std::tuple<int, int> A_shape, Eigen::VectorXf B);