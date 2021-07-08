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
      std::numeric_limits<float>().max(); // Difference between a given point
                                          // and the current middle point
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
    return ceilf(
        log2f(tensor[std::min<float>((int)round(overflow_rate * tensor_numel),
                                     tensor_numel - 1)]
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

void set_average(at::Tensor tensor, float *input_tensor_ptr) {
  float mean_value = at::flatten(tensor).mean().item<float>();
#pragma omp parallel for
  for (int i = 0; i < tensor.numel(); i++) {
    input_tensor_ptr[i] = mean_value;
  }
}

void parse_min_max(float *min, float *max) {
  if (isnan(*min)) {
    *min = NULL;
  }
  if (isnan(*max)) {
    *max = NULL;
  }
}

void quantize(at::Tensor tensor, int n_quant_levels, float min = NULL,
              float max = NULL) {
  parse_min_max(&min, &max);
  float *input_tensor_ptr = tensor.data_ptr<float>();
  if (n_quant_levels == 1) {
    set_average(tensor, input_tensor_ptr);
    return;
  }
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
}

void quantize(at::Tensor tensor, int n_quant_levels, at::Tensor min,
              at::Tensor max) {
  float *input_tensor_ptr = tensor.data_ptr<float>();
  if (n_quant_levels == 1) {
    set_average(tensor, input_tensor_ptr);
    return;
  }
  float *min_ptr = min.data_ptr<float>();
  float *max_ptr = max.data_ptr<float>();

#pragma omp parallel for
  for (int i = 0; i < tensor.numel(); i += 1) {
    torch::Tensor quant_levels =
        at::linspace(min_ptr[i], max_ptr[i], n_quant_levels);
    quantize_element(input_tensor_ptr, i, quant_levels.data_ptr<float>(),
                     n_quant_levels);
  }
}

void quantize(at::Tensor tensor, int bits, float overflow_rate,
              int quant_method = 0, float min = NULL, float max = NULL) {
  parse_min_max(&min, &max);
  float *input_tensor_ptr = tensor.data_ptr<float>();
  float *quantized_tensor_ptr = nullptr;
  if ((int)at::numel(std::get<0>(at::unique_consecutive(tensor))) == 1) {
    return;
  } else {
    if (bits == 1) {
      set_average(tensor, input_tensor_ptr);
      return;
    } else {
      if (min != NULL) {
        tensor = at::clamp_min(tensor, min);
      }
      if (max != NULL) {
        tensor = at::clamp_max(tensor, max);
      }
      if ((quant_method == 0) || (quant_method == 1)) {
        if (quant_method == 0) {
          // linear
          at::Tensor quantized_tensor = linear_quantize(
              tensor, det_sf(tensor, bits, overflow_rate, min, max), bits,
              overflow_rate);
          float *quantized_tensor_ptr = quantized_tensor.data_ptr<float>();
#pragma omp parallel for
          for (int i = 0; i < tensor.numel(); i++) {
            input_tensor_ptr[i] = quantized_tensor_ptr[i];
          }
        } else {
          // log
          at::Tensor s = at::sign(tensor);
          float sf = det_sf(tensor, bits, overflow_rate, min, max);
          tensor = at::log(at::abs(tensor)).clamp_min_(1e-20f);
          at::Tensor quantized_tensor =
              at::exp(linear_quantize(tensor, sf, bits - 1, overflow_rate)) * s;
          float *quantized_tensor_ptr = quantized_tensor.data_ptr<float>();
#pragma omp parallel for
          for (int i = 0; i < tensor.numel(); i++) {
            input_tensor_ptr[i] = quantized_tensor_ptr[i];
          }
        }
      } else {
        throw std::invalid_argument(
            "Invalid quant_method: 0 -> linear, 1 -> log.");
      }
    }
  }
}

void quantize_bindings(py::module_ &m) {
  // Binding for void quantize(at::Tensor tensor, int n_quant_levels, float min
  // = NULL, float max = NULL)
  m.def(
      "quantize",
      [](at::Tensor tensor, int n_quant_levels, float min, float max) {
        return quantize(tensor, n_quant_levels, min, max);
      },
      py::arg("tensor"), py::arg("n_quant_levels"), py::arg("min") = NULL,
      py::arg("max") = NULL);
  // Binding for void quantize(at::Tensor tensor, int n_quant_levels, at::Tensor
  // min, at::Tensor max)
  m.def(
      "quantize",
      [](at::Tensor tensor, int n_quant_levels, at::Tensor min,
         at::Tensor max) { return quantize(tensor, n_quant_levels, min, max); },
      py::arg("tensor"), py::arg("n_quant_levels"), py::arg("min"),
      py::arg("max"));
  // Bindings for void quantize(at::Tensor tensor, int bits, float
  // overflow_rate, int quant_method = 0, float min = NULL, float max = NULL)
  m.def(
      "quantize",
      [](at::Tensor tensor, int bits, float overflow_rate, int quant_method,
         float min, float max) {
        return quantize(tensor, bits, overflow_rate, quant_method, min, max);
      },
      py::arg("tensor"), py::arg("bits"), py::arg("overflow_rate"),
      py::arg("quant_method") = 0, py::arg("min") = NULL,
      py::arg("max") = NULL);
  m.def(
      "quantize",
      [](at::Tensor tensor, int bits, float overflow_rate, int quant_method,
         float min, float max) {
        return quantize(tensor, bits, overflow_rate, quant_method, min, max);
      },
      py::arg("tensor"), py::arg("bits"), py::arg("overflow_rate") = 0.,
      py::arg("quant_method") = 0, py::arg("min") = NULL,
      py::arg("max") = NULL);
}