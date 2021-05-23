#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

float det_integral(at::Tensor tensor, float overflow_rate, float min,
                   float max) {
  while (true) {
    if (overflow_rate > 1.0) {
      throw std::invalid_argument("Invalid overflow_rate value.");
    } else {
      tensor = std::get<0>(at::sort(at::flatten(at::abs(tensor)), -1, true));
      // std::cout << tensor << std::endl; // Evenly spaced...
      int64_t tensor_numel = tensor.numel();
      if ((min != NULL) && (tensor[tensor_numel - 1].item<float>() >= min)) {
        tensor[tensor_numel - 1] = min;
      }
      if ((max != NULL) && (tensor[0].item<float>() <= max)) {
        tensor[0] = max;
      }
      std::cout << tensor << std::endl;
      return ceilf(log2f(
          tensor[std::min((int)overflow_rate * tensor_numel, tensor_numel - 1)]
              .item<float>() +
          (float)1e-12));
    }
  }
}

float det_sf(at::Tensor tensor, int bits, float overflow_rate, float min,
             float max) {
  return 1 - bits + det_integral(tensor, overflow_rate, min, max);
}

at::Tensor linear_quantize(at::Tensor tensor, float sf, int bits,
                           float overflow_rate) {
  float delta = powf(2.0f, sf);
  float bound = powf(2.0f, bits - 1);
  std::cout << delta << " " << (-bound) * delta << " " << (bound - 1) * delta
            << std::endl;
  return at::clamp(at::floor(tensor / powf(2.0f, sf) + 0.5), -bound,
                   bound - 1) *
         powf(2.0f, sf);
}

void quant(at::Tensor tensor, int bits, float overflow_rate,
           int quant_method = 0, float min = NULL, float max = NULL) {
  if ((int)at::numel(std::get<0>(at::unique_consecutive(tensor))) == 1) {
    return;
  } else {
    float *input_tensor_ptr = tensor.data_ptr<float>();
    if (bits == 1) {
      float mean_value = at::flatten(tensor).mean().item<float>();
#pragma omp parallel for
      for (int i = 0; i < tensor.numel(); i++) {
        input_tensor_ptr[i] = mean_value;
      }
      return;
    } else {
      if (min != NULL) {
        tensor = at::clamp_min(tensor, min);
      }
      if (max != NULL) {
        tensor = at::clamp_max(tensor, max);
      }
      if (quant_method == 0) {
        // linear
        tensor = linear_quantize(tensor,
                                 det_sf(tensor, bits, overflow_rate, min, max),
                                 bits, overflow_rate);
        std::cout << tensor << std::endl;
      } else if (quant_method == 1) {
        // log
        at::Tensor s = at::sign(tensor);
        tensor = at::log(at::clamp_max(at::abs(tensor), 1e-20f));
        tensor = at::exp(linear_quantize(
                     tensor, det_sf(tensor, bits, overflow_rate, min, max),
                     bits - 1, overflow_rate)) *
                 s;
      } else if (quant_method == 2) {
        // tanh
        float n = powf(2.0, bits) - 1.0f;
        at::Tensor v =
            2 * (at::floor(((at::tanh(tensor) + 1.0f) / 2.0f) * n + 0.5f) / n) -
            1.0f;
        tensor = 0.5 * at::log((1.0f + v) / (1.0f - v));
      } else {
        throw std::invalid_argument(
            "Invalid quant_method: 0 -> linear, 1 -> log, 2 -> tanh.");
      }
    }
    // std::cout << tensor << std::endl; // Debugging.
#pragma omp parallel for
    float *tensor_ptr = tensor.data_ptr<float>();
    for (int i = 0; i < tensor.numel(); i++) {
      input_tensor_ptr[i] = tensor_ptr[i];
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "quantize",
      [](at::Tensor tensor, int bits, float overflow_rate, int quant_method,
         float min, float max) {
        return quant(tensor, bits, overflow_rate, quant_method, min, max);
      },
      py::arg("tensor"), py::arg("bits"), py::arg("overflow_rate"),
      py::arg("quant_method") = 0, py::arg("min") = NULL,
      py::arg("max") = NULL);
}