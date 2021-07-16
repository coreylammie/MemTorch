#define cudaSafeCall(call)                                                     \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__             \
                << "): " << cudaGetErrorString(err);                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <class T> __host__ __device__ T min_(T a, T b) {
  return !(b < a) ? a : b;
};

template <class T> __host__ __device__ T max_(T a, T b) {
  return (a < b) ? b : a;
};

template <class T> __host__ __device__ T clamp_(T x, T min, T max) {
  if (x < min)
    x = min;
  if (x > max)
    x = max;
  return x;
}

template <class T> __host__ __device__ T sign_(T x) {
  if (x > (T)0)
    return 1;
  if (x < (T)0)
    return -1;
  return (T)0.0;
}

template <class T> __host__ __device__ T abs_(T x) {
  if (x < 0)
    return -x;
  if (x >= 0)
    return x;
}

template <class T> __device__ void sort_(T *tensor, int tensor_numel) {
  T temp;
#pragma omp parallel for
  for (int i = 0; i < tensor_numel; i++) {
    for (int j = i + 1; j < tensor_numel; j++) {
      if (tensor[i] < tensor[j]) {
        temp = tensor[i];
        tensor[i] = tensor[j];
        tensor[j] = temp;
      }
    }
  }
}

__device__ int transform_2d_index(int x, int y, int len_y) {
  return x * len_y + y;
}

__device__ int transform_3d_index(int x, int y, int z, int len_y, int len_z) {
  return x * len_y * len_z + y * len_z + z;
}

int ceil_int_div(int a, int b) { return (a + b - 1) / b; }