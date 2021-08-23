#include <Eigen/Core>
#include <Eigen/SparseCore>

void solve_passive_bindings(py::module_ &m);

Eigen::VectorXf solve_sparse_linear(Eigen::SparseMatrix<float> A,
                                    Eigen::VectorXf B);

#define cudaSafeCall(call)                                                     \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__             \
                << "): " << cudaGetErrorString(err);                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int ceil_int_div(int a, int b);
