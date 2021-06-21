#include <algorithm>
constexpr int CUDA_NUM_THREADS = 128;
constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int GET_BLOCKS(const int N) {
  return std::max(std::min((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, MAXIMUM_NUM_BLOCKS), 1);
}
