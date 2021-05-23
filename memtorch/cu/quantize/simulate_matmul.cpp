#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

// CUDA kernels
// TBD //

void simulate_matmul(at::Tensor input, at::Tensor crossbar_conductance_values,
                     at::Tensor tiles_map, int *crossbar_shape,
                     float max_input_voltage, int ADC_resolution,
                     float ADC_overflow_rate, int quant_method) {
  // TBD;
}