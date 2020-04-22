#include <torch/types.h>
#include <iostream>
#include <math.h>
#include <limits>
#include "cuda_runtime.h"
#include "gpu.cuh"
#include <ATen/cuda/CUDAContext.h>

__device__ float quantize_element(float element, float* quant_levels, int num_quant_levels) {
  int middle_point; // Middle point
  int optimal_point = 0; // Optimal point
  int l = 0; // Lower bound
  int h = num_quant_levels; // Higher bound
  float difference = 1.0f; // Difference between a given point and the current middle point
  while (l <= h) {
    middle_point = l + (h - l) / 2;
    if (abs(element - quant_levels[middle_point]) < difference) {
      difference = abs(element - quant_levels[middle_point]);
      optimal_point = middle_point;
    }
    if (quant_levels[middle_point] < element) {
      l = middle_point + 1;
    } else {
      h = middle_point - 1;
    }
  }
  return quant_levels[optimal_point];
}

__global__ void quantize(int num_quant_levels,
                         float* quant_levels,
                         int num_elements,
                         float* tensor) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < num_elements; i += stride) {
    tensor[i] = quantize_element(tensor[i], quant_levels, num_quant_levels);
	}
}

__global__ void quantize_(int num_quant_levels,
                         float* min_values,
                         float* max_values,
                         int num_elements,
                         float* tensor) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

  float* quant_levels = new float[num_quant_levels];
  for (int i = index; i < num_elements; i += stride) {
    // Manually generate linspace vectors
    float step_size = (max_values[i] - min_values[i]) / (num_quant_levels - 1);
    for (int j = 0; j < num_quant_levels; ++j) {
      quant_levels[j] = min_values[i] + j * step_size;
    }
    quant_levels[num_quant_levels - 1] = max_values[i];
    tensor[i] = quantize_element(tensor[i], quant_levels, num_quant_levels);
  }
  free(quant_levels);
}

void quant_cuda(at::Tensor tensor, int num_quant_levels, float min_value, float max_value) {
  torch::Tensor quant_levels = at::linspace(min_value, max_value, num_quant_levels);
  float *quant_levels_gpu;
  cudaMalloc(&quant_levels_gpu, sizeof(float) * quant_levels.numel());
  cudaMemcpy(quant_levels_gpu, quant_levels.data<float>(), sizeof(float) * quant_levels.numel(), cudaMemcpyHostToDevice);
  quantize<<<GET_BLOCKS(tensor.numel()), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(num_quant_levels, quant_levels_gpu, tensor.numel(), tensor.data<float>());
  cudaDeviceSynchronize();
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  cudaFree(quant_levels_gpu);
}

void quant_cuda(at::Tensor tensor, int num_quant_levels, at::Tensor min_values, at::Tensor max_values) {
  quantize_<<<GET_BLOCKS(tensor.numel()), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(num_quant_levels, min_values.data<float>(), max_values.data<float>(), tensor.numel(), tensor.data<float>());
  cudaDeviceSynchronize();
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
}
