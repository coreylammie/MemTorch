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

__constant__ int NX;
__constant__ int NY;
__constant__ int NZ;

__constant__ float tsr_global; //time series resolution
__constant__ float r_off_global;
__constant__ float r_on_global;
__constant__ float pulse_dur_global;   //time series per pulse (pd_dived)
__constant__ float writing_tol_global; //rel_tol
__constant__ float res_adjustment;
__constant__ float res_adjustment_rel_tol;

//Data_Driven Global variables
__constant__ float t_n_global;
__constant__ float t_p_global;

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

//Linear Ion Drift Global variables


/**
 * Generates a programming_signal based on the time_series_resolution
 *
 * @param values Container whose values are summed.
 * @return array of time values ranging from 0 to the end of the signal
 */

__global__ void simulate_device_dd(float *device_matrix, int current_i, int current_j, int *instruction_array, float *r_n_arr, float *s_n_arr, float *r_n_half_arr, float *s_n_half_arr, float *r_p_arr, float *s_p_arr, float *r_p_half_arr, float *s_p_half_arr)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < i; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < j; j++)
  int k = threadIdx.z + blockIdx.z * blockDim.z; // for (int k = 0; k < z; k++)
  if (i < NX && j < NY && k < NZ)
  {
    float resistance_;
    int index = (k * NX * NY) + (j * NX) + i;
    if (i == current_i && j == current_j) //if it is the device to program
    {
      //printf("From Cuda: %d\n", index);
      //printf("Test : %f\n", device_matrix[0]);
      float resistance_;
      float R0 = 1 / device_matrix[index];
      //printf("Cuda old resistance main res: %f\n", R0);
      if (instruction_array[k] == 1)
      {
        resistance_ = (R0 + (s_p_arr[k] * r_p_arr[k] * (r_p_arr[k] - R0)) * pulse_dur_global) / (1 + s_p_arr[k] * (r_p_arr[k] - R0) * pulse_dur_global);
      }
      else if (instruction_array[k] == 2)
      {
        resistance_ = (R0 + (s_n_arr[k] * r_n_arr[k] * (r_n_arr[k] - R0)) * pulse_dur_global) / (1 + s_n_arr[k] * (r_n_arr[k] - R0) * pulse_dur_global);
      }
      //printf("Cuda new resistance main res: %f\n", resistance_);
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
                resistance_ += res_adjustment;
            else if(instruction_array[k] == 1 && resistance_ > r_on_global)
                resistance_ -= res_adjustment;
        }
        device_matrix[index] = 1 / resistance_;
      }
    }
    else if (i == current_i || j == current_j) //if the device is in the same row or column
    {
      int index = (k * NX * NY) + (j * NX) + i;
      //printf("From Cuda: %d\n", index);
      float R0 = 1 / device_matrix[index];
      //printf("Cuda old resistance secondary: %f\n", R0);
      if (instruction_array[k] == 1)
      {
        resistance_ = (R0 + (s_p_half_arr[k] * r_p_half_arr[k] * (r_p_half_arr[k] - R0)) * pulse_dur_global) / (1 + s_p_half_arr[k] * (r_p_half_arr[k] - R0) * pulse_dur_global);
      }
      else if (instruction_array[k] == 2)
      {
        resistance_ = (R0 + (s_n_half_arr[k] * r_n_half_arr[k] * (r_n_half_arr[k] - R0)) * pulse_dur_global) / (1 + s_n_half_arr[k] * (r_n_half_arr[k] - R0) * pulse_dur_global);
      }
      //printf("Cuda new resistance secondary: %f\n", resistance_);
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
        device_matrix[index] = 1 / resistance_;
      }
    }
  }
}
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
                resistance_ += res_adjustment;
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
                resistance_ -= res_adjustment;
              }
              if(pos_voltage_level < force_adjustment_pos_voltage_threshold){
                  pos_voltage_level += 0.02;
              }
              if(resistance_ >= R0 - res_adjustment_rel_tol*R0 && resistance_ <= R0 + res_adjustment_rel_tol*R0){
                resistance_ -= res_adjustment;
              }
              R0 = resistance_;
            }
    }
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

int countOccurrences(int arr[], int n, int x)
{
  int res = 0;
  for (int i = 0; i < n; i++)
    if (x == arr[i])
      res++;
  return res;
}

at::Tensor buildTensorFromArray(at::Tensor device_matrix, float *device_matrix_accessor, int nx, int ny, int nz)
{
  for (int i = 0; i < nx; i++)
  {
    for (int j = 0; j < ny; j++)
    {
      for (int k = 0; k < nz; k++)
      {
        device_matrix[k][i][j] = device_matrix_accessor[i + j * nx + k * nx * ny];
        printf("device matrix build tensor: %f\n",device_matrix[k][i][j]);
      }
    }
  }
  return device_matrix;
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
                               float k_p, float k_n, std::vector<float> r_p, std::vector<float> r_n, float a_p, float a_n, float b_p, float b_n, bool sim_neighbors)
{

  assert(at::cuda::is_available());
  //Assign global variables their value
  printf("index last element: %f\n",rel_tol);
  float original_pos_voltage = pos_voltage_level;
  float original_neg_voltage = neg_voltage_level;
  const size_t sz = sizeof(float);
  const size_t si = sizeof(int);
  float res_adjust = 1/force_adjustment;
  cudaSafeCall(cudaMemcpyToSymbol(res_adjustment_rel_tol, &force_adjustment_rel_tol, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(res_adjustment, &res_adjust, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(writing_tol_global, &rel_tol, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(tsr_global, &time_series_resolution, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(r_off_global, &r_off, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(r_on_global, &r_on, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(t_p_global, &t_p, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(t_n_global, &t_n, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(A_p_global, &A_p, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(A_n_global, &A_n, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(pulse_dur_global, &pulse_duration, sz, size_t(0), cudaMemcpyHostToDevice));
  //conductance_matrix = conductance_matrix.to(torch::Device("cuda:0"));
  //device_matrix = device_matrix.to(torch::Device("cuda:0"));
  float *device_matrix_accessor = device_matrix.data_ptr<float>();
  float *conductance_matrix_accessor = conductance_matrix.data_ptr<float>();
  float *device_matrix_device;
  float *conductance_matrix_device;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int nz = conductance_matrix.sizes()[0]; //n_tiles
  const int ny = conductance_matrix.sizes()[2]; //n_columns
  const int nx = conductance_matrix.sizes()[1]; //n_rows
  int max_index = ((nz - 1) * nx * ny) + ((ny - 1) * nx) + (nx - 1);
  printf("index last element: %d\n", max_index);
  printf("last element = %f\n", device_matrix_accessor[max_index]);
  int max_threads = prop.maxThreadsDim[0];
  printf("max thread: %d\n", max_threads);
  dim3 grid;
  dim3 block;
  at::Tensor new_device_matrix;
  if (nx * ny * nz > max_threads)
  {
    int n_grid = ceil_int_div(nx * ny * nz, max_threads);
    //printf("number of grids: %d\n", n_grid);

    grid = dim3(n_grid, n_grid, n_grid);
    block = dim3(ceil_int_div(nx, n_grid), ceil_int_div(ny, n_grid), ceil_int_div(nz, n_grid));
  }
  else
  {
    grid = dim3(1, 1, 1);
    block = dim3(nx, ny, nz);
  }

  //printf("\n%d\n", max_threads);
  int ndim = conductance_matrix.dim();
  //printf("ndim = %d\n", ndim);
  //CUDA accessors for the conductance matrices
  //torch::PackedTensorAccessor32<float, 3> conductance_matrix_accessor =
  //    conductance_matrix.packed_accessor32<float, 3>();
  // Data_driven specific parameters
  cudaMemcpyToSymbol(NX, &nx, si, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(NY, &ny, si, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(NZ, &nz, si, size_t(0), cudaMemcpyHostToDevice);
 // printf("\n");
 // printf("%d\n", nx);
 // printf("%d\n", ny);
 // printf("%d\n", nz);
 // printf("\n");
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
  { // This assumes symmetrical crossbars
    //to program neighbours
    printf("beginning simulate neighbours routine\n");
    int iterations = 0;
    float *neg_voltage_levels;
    float *pos_voltage_levels;
    neg_voltage_levels = new float[nz];
    pos_voltage_levels = new float[nz];
    for (int k = 0; k < nz; k++)
    {
      neg_voltage_levels[k] = original_neg_voltage;
      pos_voltage_levels[k] = original_pos_voltage;
    }

    int *instruction_array;
    instruction_array = (int *)malloc(nz * si); //vector passed to the threads to determine the polarity of the voltage: 0 -> no voltage, 1 -> positive_voltage, 2 -> negative_voltage
    float *r_n_array;
    float *r_p_array;
    float *s_n_array;
    float *s_p_array;
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
    int occurence;
    float *r_n_arr;
    float *r_p_arr;
    float *s_n_arr;
    float *s_p_arr;
    float *r_n_half_arr;
    float *r_p_half_arr;
    float *s_n_half_arr;
    float *s_p_half_arr;
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
     // printf("i = %d\n", i);
      for (int j = 0; j < ny; j++)
      { //for all j columns
     //   printf("j = %d\n", j);
        all_tiles_programmed = false;
        iterations = 0;
        while (!all_tiles_programmed)
        {
          if (iterations == 1000)
          { //Safety to ensure we do not get stuck with devices
            printf("it broke\n");
            break;
          }
          iterations++;
          for (int k = 0; k < nz; k++)
          { //will not be a very big array (corresponds to the number of tiles)
            int index = i + j * nx + k * nx * ny;
            int test_index = 0;
            //printf("index: %d\n", index);
            //printf("current: %f\n",1/device_matrix_accessor[index]);
            //printf("target: %f\n",1/conductance_matrix[k][i][j].item<float>());
            for(int jo =0; jo < nx; jo++){
              for(int ji =0; ji < ny; ji++){
                test_index = jo + ji * nx + k * nx * ny;
               // printf("device_matrix: %f\n", 1/device_matrix_accessor[test_index]);
              }
            }
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
            }
          }
          occurence = countOccurrences(instruction_array, nz, 0);
          if (occurence == nz && nz != 0)
          {
            int index = i + j * nx + 0 * nx * ny;
            all_tiles_programmed = true;
           // printf("made it to all programmed\n");
           // printf("device_accessor res: %f\n", 1/device_matrix_accessor[index]);
           // printf("conductance_matrix res: %f\n", 1/conductance_matrix[0][i][j].item<float>());
            break;
          }
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

          //for (int p = nz - 1; p >= 0; p--){
          //  printf("instruction_array: %d\n", instruction_array[p]);
          //  printf("neg_voltage_level: %f\n", neg_voltage_levels[p]);}
          //printf("Simulate device time with voltage :\n");
          simulate_device_dd<<<grid, block>>>(device_matrix_device, i, j, i_a, r_n_arr, s_n_arr, r_n_half_arr, s_n_half_arr, r_p_arr, s_p_arr, r_p_half_arr, s_p_half_arr);
          cudaSafeCall(cudaDeviceSynchronize()); //Erreur ici
          cudaMemcpy(device_matrix_accessor, device_matrix_device, nz * ny * nx * sizeof(float), cudaMemcpyDeviceToHost);
        }
      }
    }
    //printf("Device matrix accessor index 0: %f\n", 1/device_matrix_accessor[0]);
    //printf("Device matrix accessor index 0: %f\n", device_matrix_accessor[0]);
    //auto options = torch::TensorOptions().dtype(at::kFloat);
    new_device_matrix = torch::from_blob(device_matrix_accessor, {nz,nx,ny},at::kFloat);
    //new_device_matrix = buildTensorFromArray(new_device_matrix,device_matrix_accessor,nx,ny,nz);
    //new_device_matrix = at::from_blob(device_matrix_accessor, {nz,nx,ny});
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
  return new_device_matrix;
}

at::Tensor simulate_passive_linearIonDrift(at::Tensor conductance_matrix, at::Tensor device_matrix, float rel_tol,
                                           float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
                                           float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
                                           float force_adjustment_neg_voltage_threshold, float time_series_resolution, float r_off, float r_on, float u_v,
                                           float d, float pos_write_threshold, float neg_write_threshold, float p, bool sim_neighbors)
{

  assert(at::cuda::is_available());
  float original_pos_voltage = pos_voltage_level;
  float original_neg_voltage = neg_voltage_level;
  const size_t sz = sizeof(float);
  const size_t si = sizeof(int);
  float res_adjust = 1/force_adjustment;
  cudaSafeCall(cudaMemcpyToSymbol(res_adjustment_rel_tol, &force_adjustment_rel_tol, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(res_adjustment, &res_adjust, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(writing_tol_global, &rel_tol, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(r_off_global, &r_off, sz, size_t(0), cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpyToSymbol(r_on_global, &r_on, sz, size_t(0), cudaMemcpyHostToDevice));
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
  int max_index = ((nz - 1) * nx * ny) + ((ny - 1) * nx) + (nx - 1);
  printf("index last element: %d\n", max_index);
  printf("last element = %f\n", device_matrix_accessor[max_index]);
  int max_threads = prop.maxThreadsDim[0];
  printf("max thread: %d\n", max_threads);
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
  int ndim = conductance_matrix.dim();
  cudaMemcpyToSymbol(NX, &nx, si, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(NY, &ny, si, size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(NZ, &nz, si, size_t(0), cudaMemcpyHostToDevice);
  bool all_tiles_programmed = false;
  if (!sim_neighbors)
  {
  }
  else{


  }
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

/*
    float s_n_half_g = A_n * (exp(abs(neg_voltage_level) / (2 * t_p)) - 1);
    float s_p_half_g = A_p * (exp(pos_voltage_level / (2 * t_p)) - 1);
    float r_p_half_g = r_p[0] + r_p[1] * pos_voltage_level / 2;
    float r_n_half_g = r_n[0] + r_n[1] * neg_voltage_level / 2;
    cudaMemcpyToSymbol("s_n_half", &s_n_half, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("s_p_half", &s_p_half, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("r_p_half", &r_p_half, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("r_n_half", &r_n_half, sz, size_t(0), cudaMemcpyHostToDevice);


    float s_n = A_n * (exp(abs(neg_voltage_level) / t_n) - 1);
    float s_p = A_p * (exp(pos_voltage_level / t_p) - 1);
    float r_p_glob = r_p[0] + r_p[1] * pos_voltage_level;
    float r_n_glob = r_n[0] + r_n[1] * neg_voltage_level;
    cudaMemcpyToSymbol("s_n_global", &s_n, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("s_p_global", &s_p, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("r_p_global", &r_p, sz, size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("r_n_global", &r_n, sz, size_t(0), cudaMemcpyHostToDevice);

*/