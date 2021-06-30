__device__ float det_integral(float *tensor, int tensor_numel,
                              float overflow_rate, float min, float max) {
  assert(overflow_rate <= 1.0);
  merge_sort(tensor, 0, tensor_numel - 1);
  if ((min != NULL) || (max != NULL)) {
    float max_bound;
    if ((min != NULL) && (max != NULL)) {
      max_bound = max_(abs(min), abs(max));
    } else if (min != NULL) {
      max_bound = abs(min);
    } else if (max != NULL) {
      max_bound = abs(max);
    }
    if (max_bound > tensor[0]) {
      tensor[0] = max_bound;
    }
  }
  return ceilf(
      log2f(tensor[(int)round(overflow_rate * tensor_numel)] + 1e-12f));
}

__device__ float det_sf_(float *tensor, int tensor_numel, int bits,
                         float overflow_rate, float min, float max) {
  return 1 - bits + det_integral(tensor, tensor_numel, overflow_rate, min, max);
}

__device__ Eigen::VectorXf linear_quantize(Eigen::VectorXf tensor, float sf,
                                           int bits, float overflow_rate) {
  float delta = powf(2.0f, sf);
  float bound = powf(2.0f, bits - 1);
  return (tensor / powf(2.0f, sf)).unaryExpr([&](float x) {
    return clamp_<float>(floorf(x + 0.5f), -bound, bound - 1) * delta;
  });
}