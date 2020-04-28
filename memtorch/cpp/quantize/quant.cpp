#include <torch/extension.h>
#include <ATen/ATen.h>

void quantize_element(float* tensor, int index, float* quant_levels, int num_quant_levels) {
  int middle_point; // Middle point
  int optimal_point = 0; // Optimal point
  int l = 0; // Lower bound
  int h = num_quant_levels; // Higher bound
  float difference = 1.0f; // Difference between a given point and the current middle point
  while (l <= h) {
    middle_point = l + (h - l) / 2;
    if (abs(tensor[index] - quant_levels[middle_point]) < difference) {
      difference = abs(tensor[index] - quant_levels[middle_point]);
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

void quant(at::Tensor tensor, int num_quant_levels, float min_value, float max_value) {
  torch::Tensor quant_levels = at::linspace(min_value, max_value, num_quant_levels);
  for (int i = 0; i < tensor.numel(); i += 1) {
    quantize_element(tensor.data_ptr<float>(), i, quant_levels.data_ptr<float>(), num_quant_levels);
  }
}

void quant(at::Tensor tensor, int num_quant_levels, at::Tensor min_values, at::Tensor max_values) {
  float* min_values_ = min_values.data_ptr<float>();
  float* max_values_ = max_values.data_ptr<float>();
  for (int i = 0; i < tensor.numel(); i += 1) {
    torch::Tensor quant_levels = at::linspace(min_values_[i], max_values_[i], num_quant_levels);
    quantize_element(tensor.data_ptr<float>(), i, quant_levels.data_ptr<float>(), num_quant_levels);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", (void (*)(at::Tensor, int, float, float)) &quant, "tbd");
  m.def("quantize", (void (*)(at::Tensor, int, at::Tensor, at::Tensor)) &quant, "tbd");
}
