__device__ float det_integral(float *tensor, int tensor_numel,
                              float overflow_rate, float min, float max) {
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

__device__ float det_sf(float *tensor, int tensor_numel, int bits,
                        float overflow_rate, float min, float max) {
  assert(overflow_rate <= 1.0);
  sort_<float>(tensor, tensor_numel);
  return 1 - bits + det_integral(tensor, tensor_numel, overflow_rate, min, max);
}

__device__ Eigen::VectorXf linear_quantize(Eigen::VectorXf tensor, float sf,
                                           int bits, float overflow_rate) {
  float delta = powf(2.0f, sf);
  float bound = powf(2.0f, bits - 1);
  return (tensor / powf(2.0f, sf)).unaryExpr([&](float x) {
    float x_ = clamp_<float>(floorf(x + 0.5f), -bound, bound - 1) * delta;
    if (isnan(x_)) {
      return 0.0f;
    } else {
      return x_;
    }
  });
}

__device__ Eigen::VectorXf quantize(Eigen::VectorXf tensor, int bits,
                                    float overflow_rate, int quant_method) {
  if (quant_method == 0) {
    // linear
    float *tensor_data = (float *)malloc(sizeof(float) * tensor.size());
    memcpy(tensor_data, tensor.data(), sizeof(float) * tensor.size());
    float sf =
        det_sf(tensor_data, tensor.size(), bits, overflow_rate, NULL, NULL);
    delete tensor_data;
    return linear_quantize(tensor, sf, bits, overflow_rate);
  } else if (quant_method == 1) {
    // log
    float *tensor_data = (float *)malloc(sizeof(float) * tensor.size());
    memcpy(tensor_data, tensor.data(), sizeof(float) * tensor.size());
    float sf =
        det_sf(tensor_data, tensor.size(), bits, overflow_rate, NULL, NULL);
    delete tensor_data;
    bool *s = (bool *)malloc(sizeof(bool) * tensor.size());
    for (int i = 0; i < tensor.size(); i++) {
      s[i] = tensor[i] >= 0.0f;
    }
    tensor = tensor.unaryExpr(
        [&](float x) { return max_<float>(logf(abs_<float>(x)), 1e-20f); });
    tensor = linear_quantize(tensor, sf, bits - 1, overflow_rate);
    for (int i = 0; i < tensor.size(); i++) {
      if (s[i]) {
        tensor[i] = expf(tensor[i]);
      } else {
        tensor[i] = -expf(tensor[i]);
      }
    }
    delete s;
    return tensor;
  } else {
    return tensor;
  }
}
