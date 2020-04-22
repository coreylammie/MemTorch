#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// CUDA kernels
void quant_cuda(at::Tensor tensor, int num_quant_levels, float min_value, float max_value);
void quant_cuda(at::Tensor tensor, int num_quant_levels, at::Tensor min_values, at::Tensor max_values);

void quant(at::Tensor tensor, int num_quant_levels, float min_value, float max_value) {
  if (at::cuda::is_available()) {
    tensor.to(torch::Device("cuda:0"));
    quant_cuda(tensor, num_quant_levels, min_value, max_value);
  } else {
    printf("To be supported.\n");
  }
}

void quant(at::Tensor tensor, int num_quant_levels, at::Tensor min_values, at::Tensor max_values) {
  if (at::cuda::is_available()) {
    assert(tensor.numel() == min_values.numel() == max_values.numel());
    tensor.to(torch::Device("cuda:0"));
    min_values.to(torch::Device("cuda:0"));
    max_values.to(torch::Device("cuda:0"));
    quant_cuda(tensor, num_quant_levels, min_values, max_values);
  } else {
    printf("To be supported.\n");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", (void (*)(at::Tensor, int, float, float)) &quant, "tbd");
  m.def("quantize", (void (*)(at::Tensor, int, at::Tensor, at::Tensor)) &quant, "tbd");
}
