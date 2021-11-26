#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <assert.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

#include "simulate_passive.h"
#include "solve_sparse_linear.h"
#include "utils.cuh"

__constant__ float positive_voltage;
__constant__ float negative_voltage;


__constant__ float tsr; //time series resolution
__constant__ float r_off;
__constant__ float r_on;
__constant__ int ts_per_pulse;  //time series per pulse (pd_dived)
__constant__ float writing_tol; //rel_tol

//Data_Driven Global variables
__constant__ float s_n_global;
__constant__ float s_p_global;
__constant__ float r_p_global;
__constant__ float r_n_global;

__constant__ float s_n_half;
__constant__ float s_p_half;
__constant__ float r_p_half;
__constant__ float r_n_half;

__constant__ float r_p_0;
__constant__ float r_p_1;
__constant__ float r_n_0;
__constant__ float r_n_1;
__constant__ float A_p_global;
__constant__ float A_n_global;
/**
 * Generates a programming_signal based on the time_series_resolution
 *
 * @param values Container whose values are summed.
 * @return array of time values ranging from 0 to the end of the signal
 */
__global__ void simulate_device_dd(torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor,at::Tensor device_matrix, int m, int n, int z, int current_i, int current_j, bool positive)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < i; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < j; j++)
  int k = threadIdx.z + blockIdx.z * blockDim.z; // for (int k = 0; k < z; k++)
  if (i < m && j < n && k < z)
  {
    float resistance_;
    if (i == current_i && j == current_j) //if it is the device to program
    {
      float R0 = device_matrix[i][j][k];
      if (R0 < conductance_matrix_accessor[i][j][k])
      {
        for (int c = 0; i < ts_per_pulse; i++)
        {
          resistance_ = (R0 + (s_p * r_p * (r_p - R0)) * tsr) / (1 + s_p * (r_p - R0) * tsr);
        }
      }
      else
      {
        for (int c = 0; i < ts_per_pulse; i++)
        {
          resistance_ = (R0 + (s_n * r_n * (r_n - R0)) * tsr) / (1 + s_n * (r_n - R0) * tsr);
        }
      }
    }
    else if (i == current_i || j == current_j) //if the device is in the same row or column
    {
      if (R0 < conductance_matrix_accessor[i][j][k])
      {
        for (int c = 0; i < ts_per_pulse; i++)
        {
          resistance_ = (R0 + (s_p_half * r_p_half * (r_p_half - R0)) * tsr) / (1 + s_p_half * (r_p_half - R0) * tsr);
        }
      }
      else
      {
        for (int c = 0; i < ts_per_pulse; i++)
        {
          resistance_ = (R0 + (s_n_half * r_n * (r_n_half - R0)) * tsr) / (1 + s_n_half * (r_n_half - R0) * tsr);
        }
      }
    }
    device_matrix[i][j][k] = resistance_;
  }
}

__global__ void simulate_device_dd_no_neighbours(torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor, at::Tensor device_matrix, int *reached_matrix, int m, int n, int z)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < i; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < j; j++)
  int k = threadIdx.z + blockIdx.z * blockDim.z; // for (int k = 0; k < z; k++)
  if (i < m && j < n && k < z)
  {
    float R0 = device_matrix[i][j][k];
    float resistance_;
    float previous_R0;
    float s_n = s_n_global;
    float s_p = s_p_global;
    float r_p = r_p_global;
    float r_n = r_n_global;
    float pos_voltage = positive_voltage;
    float n_voltage = negative_voltage;
    int iteration = 0;
    while (conductance_matrix_accessor[i][j][k] < conductance_matrix_accessor[i][j][k] - writing_tol || conductance_matrix_accessor[i][j][k] > conductance_matrix_accessor[i][j][k] + writing_tol)
    {
      if (iteration == 100)
      {
        break;
      }
      iteration += 1;
      if (conductance_matrix_accessor[i][j][k] < conductance_matrix_accessor[i][j][k] - writing_tol)
      {
        previous_R0 = R0;
        for (int c = 0; i < ts_per_pulse; i++)
        {
          resistance_ = (R0 + (s_p * r_p * (r_p - R0)) * tsr) / (1 + s_p * (r_p - R0) * tsr);
          if (resistance_ > r_p)
          {
            R0 = max(min(resistance_, r_off), r_on); // Artificially confine the resistance between r_on and r_off
          }
        }
        if (R0 == previous_R0)
        {
          pos_voltage += 0.02; //To simulate pulsed programming
          r_p = r_p_0 + r_p_1 * pos_voltage;
          s_p = A_p_global * (exp(pos_voltage / tsr) - 1);
        }
      }
      if (conductance_matrix_accessor[i][j][k] > conductance_matrix_accessor[i][j][k] + writing_tol)
      {
        for (int c = 0; i < ts_per_pulse; i++)
        {
          resistance_ = (R0 + (s_n * r_n * (r_n - R0)) * tsr) / (1 + s_n * (r_n - R0) * tsr);
          if resistance_
            < r_n
            {
              R0 = max(min(resistance_, r_off), r_on);
            }
        }
        if (R0 == previous_R0)
        {
          n_voltage -= 0.02;
          r_n = r_n_0 + r_n_1 * n_voltage;
          s_n = A_n_global * (exp(n_voltage / tsr) - 1);
        }
      }
    }
    device_matrix[i][j][k] = R0;
  }
}

/**
 * Generates a programming_signal based on the time_series_resolution
 *
 * @param values Container whose values are summed.
 * @return array of time values ranging from 0 to the end of the signal
 */
at::Tensor simulate_passive_dd(at::Tensor conductance_matrix, at::Tensor device_matrix, float rel_tol,
                               float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
                               float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
                               float force_adjustment_neg_voltage_threshold, float time_series_resolution, float r_off, float r_on, float A_p, float A_n, float t_p, float t_n,
                               float k_p, float k_n, std::list<double> r_p, std::list<double> r_n, float a_p, float a_n, float b_p, float b_n, bool sim_neighbors)
{

  assert(at::cuda::is_available());
  //Assign global variables their value
  const size_t sz = sizeof(float;)
      cudaMemcpyToSymbol("writing_tol", &rel_tol, sz, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("tsr", &time_series_resolution, sz, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("r_off", &r_off, sz, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("r_on", &r_on, sz, size_t(0), cudaMemcpyHostToDevice);

  writing_tol = rel_tol;
  conductance_matrix = conductance_matrix.to(torch::Device("cuda:0"));
  device_matrix = device_matrix.to(torch::Device("cuda:0"));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads = prop.maxThreadsDim[0];
  dim3 grid;
  dim3 block;
  printf("\n%d\n", max_threads);

  float pd_dived = pulse_duration / time_series_resolution;
  cudaMemcpyToSymbol("ts_per_pulse", &round(pd_dived), sizeof(int), size_t(0), cudaMemcpyHostToDevice); //maybe error here (round)
  float rp_dived = refactory_period / time_series_resolution;
  // verify that pulse_duration and refactory_period is divisible by time_series_resolution.
  assert(abs(pd_dived - round(pulse_duration / time_series_resolution)) <= 1e-9);
  assert(abs(rp_dived - round(refactory_period / time_series_resolution)) <= 1e-9);
  int ndim = conductance_matrix.dim();
  printf("ndim = %d\n", ndim);
  float voltage_level;
  //CUDA accessors for the conductance matrices
  torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor =
      conductance_matrix.packed_accessor32<float, 2>();
  torch::PackedTensorAccessor32<float, 2> device_matrix_accessor =
      device_matrix.packed_accessor32<float, 2>();

  // Data_driven specific parameters
  float s_n = A_n * (exp(abs(neg_voltage_level) / t_p) - 1);
  float s_p = A_p * (exp(pos_voltage_level / t_p) - 1);
  float r_p = r_p[0] + r_p[1] * pos_voltage_level;
  float r_n = r_n[0] + r_n[1] * neg_voltage_level;
  cudaMemcpyToSymbol("s_n_global", &s_n, sz, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("s_p_global", &s_p, sz, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("r_p_global", &r_p, sz, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("r_n_global", &r_n, sz, size_t(0), cudaMemcpyHostToDevice);

  bool positive; //boolean to inform the thread of the intent of LTP (true) or LTD (false)
  int nx = conductance_matrix.sizes()[0]; //n_rows
  int ny = conductance_matrix.sizes()[1]; //n_columns
  int nz = conductance_matrix.sizes()[2]; //n_tiles
  if (!sim_neighbors)
  {
  }
  else
  { // This assumes symmetrical crossbars

    //to program neighbours
    s_n_half = A_n * (exp(abs(neg_voltage_level) / (2 * t_p)) - 1);
    s_p_half = A_p * (exp(pos_voltage_level / (2 * t_p)) - 1);
    r_p_half = r_p[0] + r_p[1] * pos_voltage_level / 2;
    r_n_half = r_n[0] + r_n[1] * neg_voltage_level / 2;
    cudaMemcpyToSymbol("s_n_half", &s_n_half, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("s_p_half", &s_p_half, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("r_p_half", &r_p_half, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("r_n_half", &r_n_half, sz, size_t(0), cudaMemcpyHostToDevice);

    int iterations = 0;
    for (int i = 0; i < nx; i++)
    { //for all i rows
      for (int j = 0; j < ny; j++)
      { //for all j columns
        while (conductance_matrix[i][j][:] < device_matrix[i][j][:] - rel_tol * device_matrix[i][j][:] || conductance_matrix[i][j][:] > device_matrix[i][j][:] + rel_tol * device_matrix[i][j][:])
        {
          if(iterations == 100){ //Safety to ensure we do not get stuck with devices
            break;
          }
          iterations++;
          if (device_matrix[i][j][k] > conductance_matrix[i][j][k])
          {
            positive = false;
          }
          else
          {
            positive = true;
          }
          simulate_device_dd(device_matrix, voltage_level, nx, ny, nz, i, j, positive)
          cudaSafeCall(cudaDeviceSynchronize());
        }
      }
    }
  }
}
printf("%f\n", force_adjustment_neg_voltage_threshold);
printf("%f\n", r_n);
return conductance_matrix;
}

at::Tensor simulate_passive_linearIonDrift(at::Tensor conductance_matrix, at::Tensor device_matrix, float rel_tol,
                                           float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
                                           float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
                                           float force_adjustment_neg_voltage_threshold, float time_series_resolution, float r_off, float r_on, float u_v,
                                           float d, float pos_write_threshold, float neg_write_threshold, float p, bool sim_neighbors)
{

  assert(at::cuda::is_available());
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads = prop.maxThreadsDim[0];
  //TODO: Implement this
  return conductance_matrix;
}

at::Tensor simulate_passive_Stanford_PKU(at::Tensor conductance_matrix, at::Tensor device_matrix, float rel_tol,
                                         float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
                                         float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
                                         float force_adjustment_neg_voltage_threshold, float time_series_resolution, float r_off, float r_on, float gap_init,
                                         float g_0, float V_0, float I_0, float read_voltage, float T_init, float R_th, float gamma_init,
                                         float beta, float t_ox, float F_min, float vel_0, float E_a, float a_0, float delta_g_init,
                                         float model_switch, float T_crit, float T_smth, bool sim_neighbors)
{

  assert(at::cuda::is_available());
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads = prop.maxThreadsDim[0];
  //TODO: Implement this
  return conductance_matrix;
}

at::Tensor simulate_passive_VTEAM(at::Tensor conductance_matrix, at::Tensor device_matrix, float rel_tol,
                                  float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
                                  float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
                                  float force_adjustment_neg_voltage_threshold, float time_series_resolution, float r_off, float r_on, float d,
                                  float k_on, float k_off, float alpha_on, float alpha_off, float v_on, float v_off, float x_on, float x_off, bool sim_neighbors)
{

  assert(at::cuda::is_available());
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads = prop.maxThreadsDim[0];
  //TODO: Implement this
  return conductance_matrix;
}