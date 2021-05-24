#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

void quantize_element(float *tensor, int index, float *quant_levels,
                      int num_quant_levels) {
  int middle_point;         // Middle point
  int optimal_point = 0;    // Optimal point
  int l = 0;                // Lower bound
  int h = num_quant_levels; // Higher bound
  float difference =
      1.0f; // Difference between a given point and the current middle point
  while (l <= h) {
    middle_point = l + (h - l) / 2;
    if (fabs(tensor[index] - quant_levels[middle_point]) < difference) {
      difference = fabs(tensor[index] - quant_levels[middle_point]);
      optimal_point = middle_point;
    }
    if (quant_levels[middle_point] < tensor[index]) {
      l = middle_point + 1;
    } else {
      h = middle_point - 1;
    }
  }
  tensor[index] = quant_levels[optimal_point];
}

float det_integral(at::Tensor tensor, float overflow_rate, float min,
                   float max) {
  if (overflow_rate > 1.0) {
    throw std::invalid_argument("Invalid overflow_rate value.");
  } else {
    tensor = std::get<0>(at::sort(at::flatten(at::abs(tensor)), -1, true));
    int64_t tensor_numel = tensor.numel();
    if ((min != NULL) || (max != NULL)) {
      float max_bound;
      if ((min != NULL) && (max != NULL)) {
        max_bound = std::max(std::abs(min), std::abs(max));
      } else if (min != NULL) {
        max_bound = std::abs(min);
      } else if (max != NULL) {
        max_bound = std::abs(max);
      }
      if (max_bound > tensor[0].item<float>()) {
        tensor[0] = max_bound;
      }
    }
    return ceilf(log2f(
        tensor[std::min((int)overflow_rate * tensor_numel, tensor_numel - 1)]
            .item<float>() +
        1e-12f));
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
  return at::clamp(at::floor(tensor / powf(2.0f, sf) + 0.5), -bound,
                   bound - 1) *
         delta;
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
        // linear (evenly-spaced)
        int n_quant_levels = floor(powf(2.0f, bits));
        if (min == NULL) {
          min = at::flatten(tensor).min().item<float>();
        }
        if (max == NULL) {
          max = at::flatten(tensor).max().item<float>();
        }
        at::Tensor quant_levels = at::linspace(min, max, n_quant_levels);
#pragma omp parallel for
        for (int i = 0; i < tensor.numel(); i++) {
          quantize_element(input_tensor_ptr, i, quant_levels.data_ptr<float>(),
                           n_quant_levels);
        }
        return;
      } else if (quant_method == 1) {
        // linear (scaled by 2^n)
        tensor = linear_quantize(tensor,
                                 det_sf(tensor, bits, overflow_rate, min, max),
                                 bits, overflow_rate);
      } else if (quant_method == 2) {
        // log
        at::Tensor s = at::sign(tensor);
        tensor = at::log(at::clamp_min(at::abs(tensor), 1e-20f));
        tensor = at::exp(linear_quantize(
                     tensor, det_sf(tensor, bits, overflow_rate, min, max),
                     bits - 1, overflow_rate)) *
                 s;
      } else if (quant_method == 3) {
        // tanh
        std::cout << bits << std::endl;
        float n = powf(2.0, bits) - 1.0f;
        float max_bound;
        if ((min != NULL) && (max != NULL)) {
          max_bound = std::max(std::abs(min), std::abs(max));
        } else if (min != NULL) {
          max_bound = std::abs(min);
        } else if (max != NULL) {
          max_bound = std::abs(max);
        }
        float max_bound_ratio =
            at::flatten(tensor).max().item<float>() / max_bound;
        at::Tensor v =
            2 * (at::floor(((at::tanh(tensor) + 1.0f) / 2.0f) * n + 0.5f) / n) -
            1.0f;
        if (max_bound_ratio < 1.0f) {
          v *= max_bound_ratio;
        }
        std::cout << v << std::endl;
        tensor = at::arctan(v);
      } else {
        throw std::invalid_argument(
            "Invalid quant_method: 0 -> linear (evenly-spaced), 1 -> linear "
            "(scaled by 2^n), 2 -> log, 3 -> tanh.");
      }
    }
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