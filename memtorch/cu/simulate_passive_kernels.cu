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

__constant__ int NX; //number of rows
__constant__ int NY; //number of columns
__constant__ int NZ; //number of tiles

__constant__ float tsr_global; //time series resolution
__constant__ float r_off_global;
__constant__ float r_on_global;
__constant__ float pulse_dur_global;   //pulse duration
__constant__ float writing_tol_global; //rel_tol
__constant__ float res_adjustment; //adjustment made if the resistance is stuck
__constant__ float res_adjustment_rel_tol;

//Data_Driven Global variables
__constant__ float t_n_global;
__constant__ float t_p_global;

__constant__ float s_n_global;
__constant__ float s_p_global;
__constant__ float r_p_global;
__constant__ float r_n_global;

__constant__ float r_p_0;
__constant__ float r_p_1;
__constant__ float r_n_0;
__constant__ float r_n_1;
__constant__ float A_p_global;
__constant__ float A_n_global;



/**
 * Cuda kernel to simulate the devices with neighbors
 *
 * @param device_matrix the device matrix of conductances
 * @param current_i the position in i for the current device to simulate
 * @param current_j the position in j for the current device to simulate
 * @param instruction_array array of integers corresponding to the current instruction for the tile being programmed
 * @param r_n_arr array of r_n values shared across devices
 * @param s_n_arr array of s_n values shared across devices
 * @param r_p_arr array of r_n values shared across devices
 * @param s_p_arr array of s_n values shared across devices
 * @param r_n_half_arr array of r_n/2 values shared across devices
 * @param s_n_half_arr array of s_n/2 values shared across devices
 * @param r_p_half_arr array of r_n/2 values shared across devices
 * @param s_p_half_arr array of s_n/2 values shared across devices
 */

__global__ void simulate_device_dd(float *device_matrix, int current_i, int current_j, int *instruction_array, float *r_n_arr, float *s_n_arr, float *r_n_half_arr, float *s_n_half_arr, float *r_p_arr, float *s_p_arr, float *r_p_half_arr, float *s_p_half_arr)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < NX; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < j; j++)
  int k = threadIdx.z + blockIdx.z * blockDim.z; // for (int k = 0; k < z; k++)
  if (i < NX && j < NY && k < NZ)
  {
    float resistance_;
    int index = (k * NX * NY) + (j * NX) + i;
    if (i == current_i && j == current_j) //if it is the device to program
    {
      float R0 = 1 / device_matrix[index];
      if (instruction_array[k] == 1)
      {
        resistance_ = (R0 + (s_p_arr[k] * r_p_arr[k] * (r_p_arr[k] - R0)) * pulse_dur_global) / (1 + s_p_arr[k] * (r_p_arr[k] - R0) * pulse_dur_global);
        if(resistance_ < r_p_arr[k]){
            resistance_ = R0;
        }
      }
      else if (instruction_array[k] == 2)
      {
        resistance_ = (R0 + (s_n_arr[k] * r_n_arr[k] * (r_n_arr[k] - R0)) * pulse_dur_global) / (1 + s_n_arr[k] * (r_n_arr[k] - R0) * pulse_dur_global);
        if(resistance_ > r_n_arr[k]){
            resistance_ = R0;
        }
      }
      if (instruction_array[k] != 0)
      {
        if (resistance_ > r_off_global)
        {
          resistance_ = r_off_global;
        }
        if (resistance_ < r_on_global)
        {
          resistance_ = r_on_global;
        }
        if(resistance_ >= R0 - res_adjustment_rel_tol*R0 && resistance_ <= R0 + res_adjustment_rel_tol*R0){
            if(instruction_array[k] == 2 && resistance_ < r_off_global)
                resistance_ += res_adjustment*resistance_;
            else if(instruction_array[k] == 1 && resistance_ > r_on_global)
                resistance_ -= res_adjustment*resistance_;
        }
        device_matrix[index] = 1 / resistance_;
      }
    }
    else if (i == current_i || j == current_j) //if the device is in the same row or column
    {
      float R0 = 1 / device_matrix[index];
      if (instruction_array[k] == 1)
      {
        resistance_ = (R0 + (s_p_half_arr[k] * r_p_half_arr[k] * (r_p_half_arr[k] - R0)) * pulse_dur_global) / (1 + s_p_half_arr[k] * (r_p_half_arr[k] - R0) * pulse_dur_global);
      }
      else if (instruction_array[k] == 2)
      {
        resistance_ = (R0 + (s_n_half_arr[k] * r_n_half_arr[k] * (r_n_half_arr[k] - R0)) * pulse_dur_global) / (1 + s_n_half_arr[k] * (r_n_half_arr[k] - R0) * pulse_dur_global);
      }
      if (instruction_array[k] != 0)
      {
        //Check to ensure that the resistance remains within possible range
        if (resistance_ > r_off_global)
        {
          resistance_ = r_off_global;
        }
        if (resistance_ < r_on_global)
        {
          resistance_ = r_on_global;
        }
        device_matrix[index] = 1 / resistance_;
      }
    }
  }
}

/**
 * Cuda kernel to simulate the devices without neighbors
 *
 * @param device_matrix the device matrix of conductances
 * @param conductance_matrix the matrix of target conductances
 * @param force_adjustment_pos_voltage_threshold maximum positive voltage
 * @param force_adjustment_neg_voltage_threshold minimum negative voltage

 */
__global__ void simulate_device_dd_no_neighbours(float *device_matrix,float *conductance_matrix,float force_adjustment_pos_voltage_threshold,float force_adjustment_neg_voltage_threshold)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < i; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < j; j++)
  int k = threadIdx.z + blockIdx.z * blockDim.z; // for (int k = 0; k < z; k++)
  if (i < NX && j < NY && k < NZ)
  {
   int index = (k * NX * NY) + (j * NX) + i;
   float R0 = 1 / device_matrix[index];
   float target_R = 1 / conductance_matrix[index];
   float resistance_;
   float s_n;
   float r_n;
   float s_p;
   float r_p;
   float neg_voltage_level = negative_voltage;
   float pos_voltage_level = positive_voltage;
   int iterations = 0;
   while ((R0 < target_R - writing_tol_global*target_R || R0 > target_R + writing_tol_global*target_R) && iterations < 1000)
    {
       iterations += 1;
       if(R0 < target_R - writing_tol_global*target_R)
            {
              s_n = A_n_global * (exp(abs(neg_voltage_level) / t_n_global) - 1);
              r_n = r_n_0 + r_n_1 * neg_voltage_level;
              resistance_ = (R0 + (s_n * r_n * (r_n - R0)) * pulse_dur_global) / (1 + s_n * (r_n - R0) * pulse_dur_global);
              if(resistance_ > r_n)
              {
                resistance_ = R0;
              }
              pos_voltage_level = positive_voltage;
              if(neg_voltage_level > force_adjustment_neg_voltage_threshold){
                    neg_voltage_level -= 0.02;
              }
              if(resistance_ >= R0 - res_adjustment_rel_tol*R0 && resistance_ <= R0 + res_adjustment_rel_tol*R0){
                resistance_ += res_adjustment*resistance_;
              }
              R0 = resistance_;
            }
       else if (R0 > target_R + writing_tol_global*target_R)
            {
              s_p = A_p_global * (exp(abs(pos_voltage_level) / t_p_global) - 1);
              r_p = r_p_0 + r_p_1 * pos_voltage_level;
              resistance_ = (R0 + (s_p * r_p * (r_p - R0)) * pulse_dur_global) / (1 + s_p * (r_p - R0) * pulse_dur_global);
              neg_voltage_level = negative_voltage;
              if (resistance_ < r_p)
              {
                resistance_ = R0; // Artificially confine the resistance between r_on and r_off
              }
              if(resistance_ >= R0 - res_adjustment_rel_tol*R0 && resistance_ <= R0 + res_adjustment_rel_tol*R0){
                resistance_ -= res_adjustment*resistance_;
              }
              if(pos_voltage_level < force_adjustment_pos_voltage_threshold){
                  pos_voltage_level += 0.02;
              }
              R0 = resistance_;
            }
    }
  //Check to ensure that the resistance remains within possible range
  if (R0 > r_off_global)
      {
          R0 = r_off_global;
      }
  if (R0 < r_on_global)
        {
          R0 = r_on_global;
        }
  device_matrix[index] = 1/R0;
}
}

/**
 * Simulates passive crossbar programming using CUDA. The amplitudes of the voltages are increased or decreased by 0.02V (TODO: make this a parameter) every time the desired resistance is not achieved
 * The voltages are reset to their initial values if the desired resistance is passed.
 *
 *
 * @param conductance_matrix
 * @param device_matrix
 * @param cuda_malloc_heap_size
 * @param rel_tol acceptable tolerance on the achieved tolerance value
 * @param pulse_duration duration of the pulses
 * @param cuda_malloc_heap_size maximum heap size for Cuda
 * @param refactory_period refactory period in between every pulse sent //Not currently used//
 * @param pos_voltage_level initial positive voltage level
 * @param neg_voltage_level initial negative voltage level
 * @param timeout timeout in between every pulse sent //Not currently used//
 * @param force_adjustment percentage of the current resistance value to artificially force towards target resistance if the programming has not changed the resistance significantly (+/- force_adjustment*current_resistance)
 * @param force_adjustment_rel_tol percentage of the previous resistance used in determining if the resistance has not changed enough in one cycle to warrant forced adjustment using parameter force_adjustment (if current_resistance </> previous_resistance +/- force_adjustment_rel_tol*previous_resistance where the signs depend on polarity )
 * @param force_adjustment_pos_voltage_threshold Maximum voltage that the incrementation of 0.02V can lead to (voltage will always be lower than this even with incrementation)
 * @param force_adjustment_neg_voltage_threshold Minimum voltage that the decrementation of 0.02V can lead to (voltage will always be higher than this even with decrementation)
 * @param time_series_resolution time series resolution used in the simulation //Not currently used//
 * @param r_off Resistance at HRS
 * @param r_on Resistance at LRS
 * @param A_p A_p parameter of Data Driven model
 * @param A_n A_n parameter of Data Driven model
 * @param t_p t_p parameter of Data Driven model
 * @param t_n t_n parameter of Data Driven model
 * @param k_p k_p parameter of Data Driven model
 * @param k_n k_n parameter of Data Driven model
 * @param r_p r_p parameter of Data Driven model
 * @param r_n r_n parameter of Data Driven model
 * @param a_p r_p parameter of Data Driven model
 * @param a_n r_n parameter of Data Driven model
 * @param b_p r_p parameter of Data Driven model
 * @param b_n r_n parameter of Data Driven model
 * @param simulate_neighbors boolean to determine if neighbor simulation is necessary
 * @return Tensor of new devices
 */
at::Tensor simulate_passive_dd(at::Tensor conductance_matrix, at::Tensor device_matrix,int cuda_malloc_heap_size, float rel_tol,
                               float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
                               float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
                               float force_adjustment_neg_voltage_threshold, float time_series_resolution, float r_off, float r_on, float A_p, float A_n, float t_p, float t_n,
                               float k_p, float k_n, std::vector<float> r_p, std::vector<float> r_n, float a_p, float a_n, float b_p, float b_n, bool sim_neighbors)
{

  assert(at::cuda::is_available());
  //Assign global variables their value
  float original_pos_voltage = pos_voltage_level;
  float original_neg_voltage = neg_voltage_level;
  const size_t sz = sizeof(float);
  const size_t si = sizeof(int);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                     size_t(1024) * size_t(1024) *
                         size_t(cuda_malloc_heap_size));
  cudaSafeCall(cudaMemcpyToSymbol(res_adjustment_rel_tol, &force_adjustment_rel_tol, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(res_adjustment, &force_adjustment, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(writing_tol_global, &rel_tol, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(tsr_global, &time_series_resolution, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(r_off_global, &r_off, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(r_on_global, &r_on, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(t_p_global, &t_p, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(t_n_global, &t_n, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(A_p_global, &A_p, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(A_n_global, &A_n, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(pulse_dur_global, &pulse_duration, sz, size_t(0), cudaMemcpyHostToDevice));
  float *device_matrix_accessor = device_matrix.data_ptr<float>();
  float *conductance_matrix_accessor = conductance_matrix.data_ptr<float>();
  float *device_matrix_device;
  float *conductance_matrix_device;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int nz = conductance_matrix.sizes()[0]; //n_tiles
  const int ny = conductance_matrix.sizes()[2]; //n_columns
  const int nx = conductance_matrix.sizes()[1]; //n_rows
  int max_threads = prop.maxThreadsDim[0];
  dim3 grid;
  dim3 block;
  at::Tensor new_device_matrix;
  if (nx * ny * nz > max_threads)
  {
    int n_grid = ceil_int_div(nx * ny * nz, max_threads);
    grid = dim3(n_grid, n_grid, n_grid);
    block = dim3(ceil_int_div(nx, n_grid), ceil_int_div(ny, n_grid), ceil_int_div(nz, n_grid));
  }
  else
  {
    grid = dim3(1, 1, 1);
    block = dim3(nx, ny, nz);
  }
  cudaMemcpyToSymbol(NX, &nx, si, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(NY, &ny, si, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(NZ, &nz, si, size_t(0), cudaMemcpyHostToDevice);
  //boolean set to true when all tiles are programmed and false otherwise
  bool all_tiles_programmed = false;
  if (!sim_neighbors)
  {
    cudaMalloc(&conductance_matrix_device, sizeof(float) * nz * nx * ny);
    cudaMemcpy(conductance_matrix_device, conductance_matrix_accessor, nz * ny * nx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&device_matrix_device, sizeof(float) * nz * nx * ny);
    cudaMemcpy(device_matrix_device, device_matrix_accessor, nz * ny * nx * sizeof(float), cudaMemcpyHostToDevice);
    simulate_device_dd_no_neighbours<<<grid, block>>>(device_matrix_device,conductance_matrix_device,force_adjustment_pos_voltage_threshold,force_adjustment_neg_voltage_threshold);
    cudaSafeCall(cudaDeviceSynchronize()); //Erreur ici
    cudaMemcpy(device_matrix_accessor, device_matrix_device, nz * ny * nx * sizeof(float), cudaMemcpyDeviceToHost);
    new_device_matrix = torch::from_blob(device_matrix_accessor, {nz,nx,ny},at::kFloat);
    cudaSafeCall(cudaFree(device_matrix_device));
    cudaSafeCall(cudaFree(conductance_matrix_device));
  }
  else
  {
    int iterations = 0;
    float *neg_voltage_levels;
    float *pos_voltage_levels;
    //vector passed to the threads to determine the amplitude of the negative voltages
    neg_voltage_levels = new float[nz];
    //vector passed to the threads to determine the amplitude of the positive voltages
    pos_voltage_levels = new float[nz];
    //set all the voltages to be passed to the threads to their initial value
    for (int k = 0; k < nz; k++)
    {
      neg_voltage_levels[k] = original_neg_voltage;
      pos_voltage_levels[k] = original_pos_voltage;
    }
    int *instruction_array;
    //vector passed to the threads to determine the polarity of the voltage on each tile: 0 -> no voltage, 1 -> positive_voltage, 2 -> negative_voltage
    //the size of this vector corresponds to the number of tiles
    instruction_array = (int *)malloc(nz * si);
    float *r_n_array;
    float *r_p_array;
    float *s_n_array;
    float *s_p_array;
    //vector of data driven parameter for each tile as they will be the same for all devices
    r_n_array = (float *)malloc(nz * sz);
    r_p_array = (float *)malloc(nz * sz);
    s_n_array = (float *)malloc(nz * sz);
    s_p_array = (float *)malloc(nz * sz);
    float *r_n_half_array;
    float *r_p_half_array;
    float *s_n_half_array;
    float *s_p_half_array;
    r_n_half_array = (float *)malloc(nz * sz);
    r_p_half_array = (float *)malloc(nz * sz);
    s_n_half_array = (float *)malloc(nz * sz);
    s_p_half_array = (float *)malloc(nz * sz);
    int *i_a;
    //n_programmed corresponds to the number of tiles programmed
    int n_programmed;
    float *r_n_arr;
    float *r_p_arr;
    float *s_n_arr;
    float *s_p_arr;
    float *r_n_half_arr;
    float *r_p_half_arr;
    float *s_n_half_arr;
    float *s_p_half_arr;
    //Allocate the memory for all necessary arrays on the GPU
    cudaMalloc(&i_a, sizeof(int) * nz);
    cudaMalloc(&r_n_arr, sizeof(float) * nz);
    cudaMalloc(&r_p_arr, sizeof(float) * nz);
    cudaMalloc(&s_n_arr, sizeof(float) * nz);
    cudaMalloc(&s_p_arr, sizeof(float) * nz);
    cudaMalloc(&r_n_half_arr, sizeof(float) * nz);
    cudaMalloc(&r_p_half_arr, sizeof(float) * nz);
    cudaMalloc(&s_n_half_arr, sizeof(float) * nz);
    cudaMalloc(&s_p_half_arr, sizeof(float) * nz);
    cudaMalloc(&device_matrix_device, sizeof(float) * nz * nx * ny);

    for (int i = 0; i < nx; i++)
    { //for all i rows
      for (int j = 0; j < ny; j++)
      { //for all j columns
        all_tiles_programmed = false;
        iterations = 0;
        while (!all_tiles_programmed)
        {
          if (iterations == 100)
          { //Safety to ensure we do not get stuck with devices TODO: make this a variable
            printf("unable to program device(s) at row %d and column %d\n",i,j);
            all_tiles_programmed = true;
            iterations = 0;
          }
          iterations++;
          n_programmed = 0;
          for (int k = 0; k < nz; k++)
          { //should not be a very big array (corresponds to the number of tiles)
            int index = i + j * nx + k * nx * ny;
            if (1/conductance_matrix[k][i][j].item<float>() - rel_tol * 1/conductance_matrix[k][i][j].item<float>() > 1/device_matrix_accessor[index])
            {
              instruction_array[k] = 2;
              s_n_array[k] = A_n * (exp(abs(neg_voltage_levels[k]) / t_n) - 1);
              r_n_array[k] = r_n[0] + r_n[1] * neg_voltage_levels[k];
              s_n_half_array[k] = A_n * (exp(abs(neg_voltage_levels[k] / 2) / t_n) - 1);
              r_n_half_array[k] = r_n[0] + r_n[1] * neg_voltage_levels[k] / 2;
              s_p_array[k] = 0;
              r_p_array[k] = 0;
              s_p_half_array[k] = 0;
              r_p_half_array[k] = 0;
              pos_voltage_levels[k] = original_pos_voltage;
              //0.02 hard coded so far
              if(neg_voltage_levels[k] > force_adjustment_neg_voltage_threshold){
                    neg_voltage_levels[k] -= 0.02;
              }
            }
            else if (1/conductance_matrix[k][i][j].item<float>() + rel_tol * 1/conductance_matrix[k][i][j].item<float>() < 1/(device_matrix_accessor[index]))
            {
              instruction_array[k] = 1;
              s_p_array[k] = A_p * (exp(abs(pos_voltage_levels[k]) / t_p) - 1);
              r_p_array[k] = r_p[0] + r_p[1] * pos_voltage_levels[k];
              s_p_half_array[k] = A_p * (exp(abs(pos_voltage_levels[k] / 2) / t_p) - 1);
              r_p_half_array[k] = r_p[0] + r_p[1] * pos_voltage_levels[k] / 2;
              s_n_array[k] = 0;
              r_n_array[k] = 0;
              s_n_half_array[k] = 0;
              r_n_half_array[k] = 0;
              neg_voltage_levels[k] = original_neg_voltage;
              if(pos_voltage_levels[k] < force_adjustment_pos_voltage_threshold){
                  pos_voltage_levels[k] += 0.02;
              }
            }
            else
            {
              instruction_array[k] = 0;
              n_programmed++;
            }
          }
          if (n_programmed == nz && nz != 0)
          {
            all_tiles_programmed = true;
            iterations = 0;
            n_programmed = 0;
          }
          //The gain in rapidity is probably limited by these transfers of data to the GPU but is still significant.
          cudaMemcpy(i_a, instruction_array, nz * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(r_n_arr, r_n_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(s_n_arr, s_n_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(r_n_half_arr, r_n_half_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(s_n_half_arr, s_n_half_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(r_p_arr, r_p_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(s_p_arr, s_p_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(r_p_half_arr, r_p_half_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(s_p_half_arr, s_p_half_array, nz * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(device_matrix_device, device_matrix_accessor, nz * ny * nx * sizeof(float), cudaMemcpyHostToDevice);
          simulate_device_dd<<<grid, block>>>(device_matrix_device, i, j, i_a, r_n_arr, s_n_arr, r_n_half_arr, s_n_half_arr, r_p_arr, s_p_arr, r_p_half_arr, s_p_half_arr);
          cudaSafeCall(cudaDeviceSynchronize());
          cudaMemcpy(device_matrix_accessor, device_matrix_device, nz * ny * nx * sizeof(float), cudaMemcpyDeviceToHost);
        }
      }
    }
    new_device_matrix = torch::from_blob(device_matrix_accessor, {nz,nx,ny},at::kFloat);
    cudaSafeCall(cudaFree(i_a));
    cudaSafeCall(cudaFree(r_n_arr));
    cudaSafeCall(cudaFree(s_n_arr));
    cudaSafeCall(cudaFree(r_n_half_arr));
    cudaSafeCall(cudaFree(s_n_half_arr));
    cudaSafeCall(cudaFree(r_p_arr));
    cudaSafeCall(cudaFree(s_p_arr));
    cudaSafeCall(cudaFree(r_p_half_arr));
    cudaSafeCall(cudaFree(s_p_half_arr));
    cudaSafeCall(cudaFree(device_matrix_device));
  }
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  return new_device_matrix;
}

at::Tensor simulate_passive_linearIonDrift(at::Tensor conductance_matrix, at::Tensor device_matrix, int cuda_malloc_heap_size, float rel_tol,
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

at::Tensor simulate_passive_Stanford_PKU(at::Tensor conductance_matrix, at::Tensor device_matrix,int cuda_malloc_heap_size, float rel_tol,
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

at::Tensor simulate_passive_VTEAM(at::Tensor conductance_matrix, at::Tensor device_matrix, int cuda_malloc_heap_size, float rel_tol,
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
