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

__device__ void merge(float *arr, int low, int high, int mid) {
  int i, j, k;
  float c[50];
  i = low;
  k = low;
  j = mid + 1;
  while (i <= mid && j <= high) {
    if (arr[i] > arr[j]) {
      c[k] = arr[i];
      k++;
      i++;
    } else {
      c[k] = arr[j];
      k++;
      j++;
    }
  }
  while (i <= mid) {
    c[k] = arr[i];
    k++;
    i++;
  }
  while (j <= high) {
    c[k] = arr[j];
    k++;
    j++;
  }
  for (i = low; i < k; i++) {
    arr[i] = c[i];
  }
}

__device__ void merge_sort(float *arr, int low, int high) {
  int mid;
  if (low < high) {
    mid = (low + high) / 2;
    merge_sort(arr, low, mid);
    merge_sort(arr, mid + 1, high);
    merge(arr, low, high, mid);
  }
}

__device__ int transform_2d_index(int x, int y, int len_y) {
  return x * len_y + y;
}

__device__ int transform_3d_index(int x, int y, int z, int len_y, int len_z) {
  return x * len_y * len_z + y * len_z + z;
}

int ceil_int_div(int a, int b) { return (a + b - 1) / b; }