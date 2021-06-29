#include "cuda_runtime.h"
#include "utils.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <torch/types.h>

#include <Eigen/Core>

float det_integral(float *tensor, int tensor_numel, float overflow_rate,
                   float min, float max) {
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
  return ceilf(log2f(tensor[min_<int>((int)round(overflow_rate * tensor_numel),
                                      tensor_numel - 1)] +
                     1e-12f));
}

float det_sf(float *tensor, int tensor_numel, int bits, float overflow_rate,
             float min, float max) {
  return 1 - bits + det_integral(tensor, tensor_numel, overflow_rate, min, max);
}

Eigen::VectorXf linear_quantize(Eigen::VectorXf tensor, float sf, int bits,
                                float overflow_rate) {
  float delta = powf(2.0f, sf);
  float bound = powf(2.0f, bits - 1);
  return (tensor / powf(2.0f, sf)).unaryExpr([&](float x) {
    return clamp_<float>(floorf(x + 0.5f), -bound, bound - 1) * delta;
  });
}

at::Tensor quantize(at::Tensor tensor, int bits, float overflow_rate,
                    int quant_method, float min, float max) {
  assert(quant_method == 0);
  float *tensor_accessor = tensor.data_ptr<float>();
  Eigen::VectorXf t =
      Eigen::Map<Eigen::VectorXf>(&tensor_accessor[0], tensor.numel());
  if ((min != NULL) && (max != NULL)) {
    t = t.unaryExpr([&](float x) { return clamp_<float>(x, min, max); });
  } else if (min != NULL) {
    t = t.unaryExpr([&](float x) { return max_<float>(x, min); });
  } else if (max != NULL) {
    t = t.unaryExpr([&](float x) { return min_<float>(x, max); });
  }
  float sf =
      det_sf(tensor_accessor, tensor.numel(), bits, overflow_rate, min, max);
  ;
  memcpy(tensor_accessor, linear_quantize(t, sf, bits, overflow_rate).data(),
         sizeof(float) * tensor.numel());
  return tensor;
}