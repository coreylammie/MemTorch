#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>

#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

#include "simulate_passive_kernels.cuh"

// Default values for r_p and r_n parameters of the Data_Driven2021 model.
std::vector<float> r_p{2699.2336, -672.930205};  // r_p voltage-dependent resistive boundary function coefficients.
std::vector<float> r_n{649.413746, -1474.32358}; // r_n voltage-dependent resistive boundary function coefficients.

void simulate_passive_bindings(py::module_ &m) {
  // Data_Driven2021 model
  m.def(
      "simulate_passive",
      [&](at::Tensor conductance_matrix, at::Tensor device_matrix,
          int cuda_malloc_heap_size, float rel_tol, float pulse_duration,
          float refactory_period, float pos_voltage_level,
          float neg_voltage_level, float timeout, float force_adjustment,
          float force_adjustment_rel_tol,
          float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold,
          float force_adjustment_voltage,
          int failure_iteration_threshold,
          float time_series_resolution, float r_off, float r_on, float A_p,
          float A_n, float t_p, float t_n, float k_p, float k_n,
          std::vector<float> r_p, std::vector<float> r_n, float a_p, float a_n,
          float b_p, float b_n, bool sim_neighbors) {
        return simulate_passive_dd(
            conductance_matrix, device_matrix, cuda_malloc_heap_size, rel_tol,
            pulse_duration, refactory_period, pos_voltage_level,
            neg_voltage_level, timeout, force_adjustment,
            force_adjustment_rel_tol, force_adjustment_pos_voltage_threshold,
            force_adjustment_neg_voltage_threshold,
            force_adjustment_voltage, failure_iteration_threshold,
            time_series_resolution,
            r_off, r_on, A_p, A_n, t_p, t_n, k_p, k_n, r_p, r_n, a_p, a_n, b_p,
            b_n, sim_neighbors);
      },
      py::arg("conductance_matrix"), py::arg("device_matrix"),
      py::arg("cuda_malloc_heap_size") = 50, py::arg("rel_tol") = 0.1,
      py::arg("pulse_duration") = 1e-3, py::arg("refactory_period") = 0,
      py::arg("pos_voltage_level") = 1.0, py::arg("neg_voltage_level") = -1.0,
      py::arg("timeout") = 5, py::arg("force_adjustment") = 1e-3,
      py::arg("force_adjustment_rel_tol") = 1e-1,
      py::arg("force_adjustment_pos_voltage_threshold") = 0,
      py::arg("force_adjustment_neg_voltage_threshold") = 0,
      py::arg("force_adjustment_voltage") = 0.2,
      py::arg("failure_iteration_threshold") = 1000,
      py::arg("time_series_resolution") = 1e-10, py::arg("r_off") = 10000,
      py::arg("r_on") = 1000, py::arg("A_p") = 600.10075,
      py::arg("A_n") = -34.5988399, py::arg("t_p") = -0.0212028,
      py::arg("t_n") = -0.05343997, py::arg("k_p") = 5.11e-4,
      py::arg("k_n") = 1.17e-3, py::arg("r_p") = r_p, py::arg("r_n") = r_n,
      py::arg("a_p") = 0.32046175, py::arg("a_n") = 0.32046175,
      py::arg("b_p") = 2.71689828, py::arg("b_n") = 2.71689828,
      py::arg("simulate_neighbours") =
          true);

  // Linear Ion Drift
  m.def(
      "simulate_passive",
      [&](at::Tensor conductance_matrix, at::Tensor device_matrix,
          int cuda_malloc_heap_size, float rel_tol, float pulse_duration,
          float refactory_period, float pos_voltage_level,
          float neg_voltage_level, float timeout, float force_adjustment,
          float force_adjustment_rel_tol,
          float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold,
          float force_adjustment_voltage,
          int failure_iteration_threshold,
          float time_series_resolution, float r_off, float r_on, float u_v,
          float d, float pos_write_threshold, float neg_write_threshold,
          float p, bool sim_neighbors) {
        return simulate_passive_linearIonDrift(
            conductance_matrix, device_matrix, cuda_malloc_heap_size, rel_tol,
            pulse_duration, refactory_period, pos_voltage_level,
            neg_voltage_level, timeout, force_adjustment,
            force_adjustment_rel_tol, force_adjustment_pos_voltage_threshold,
            force_adjustment_neg_voltage_threshold,
            force_adjustment_voltage, failure_iteration_threshold,
            time_series_resolution,
            r_off, r_on, u_v, d, pos_write_threshold, neg_write_threshold, p,
            sim_neighbors);
      },
      py::arg("conductance_matrix"), py::arg("device_matrix"),
      py::arg("cuda_malloc_heap_size") = 50, py::arg("rel_tol") = 0.1,
      py::arg("pulse_duration") = 1e-3, py::arg("refactory_period") = 0,
      py::arg("pos_voltage_level") = 1.0, py::arg("neg_voltage_level") = -1.0,
      py::arg("timeout") = 5, py::arg("force_adjustment") = 1e-3,
      py::arg("force_adjustment_rel_tol") = 1e-1,
      py::arg("force_adjustment_pos_voltage_threshold") = 0,
      py::arg("force_adjustment_neg_voltage_threshold") = 0,
      py::arg("force_adjustment_voltage") = 0.2,
      py::arg("failure_iteration_threshold") = 1000,
      py::arg("time_series_resolution") = 1e-4, py::arg("r_off") = 10000,
      py::arg("r_on") = 1000, py::arg("u_v") = 1e-14, py::arg("d") = 10e-9,
      py::arg("pos_write_threshold") = 0.55,
      py::arg("neg_write_threshold") = -0.55, py::arg("p") = 1,
      py::arg("simulate_neighbours") = true);

  // VTEAM
  m.def(
      "simulate_passive",
      [&](at::Tensor conductance_matrix, at::Tensor device_matrix,
          int cuda_malloc_heap_size, float rel_tol, float pulse_duration,
          float refactory_period, float pos_voltage_level,
          float neg_voltage_level, float timeout, float force_adjustment,
          float force_adjustment_rel_tol,
          float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold,
          float force_adjustment_voltage,
          int failure_iteration_threshold,
          float time_series_resolution, float r_off, float r_on, float d,
          float k_on, float k_off, float alpha_on, float alpha_off, float v_on,
          float v_off, float x_on, float x_off, bool sim_neighbors) {
        return simulate_passive_VTEAM(
            conductance_matrix, device_matrix, cuda_malloc_heap_size, rel_tol,
            pulse_duration, refactory_period, pos_voltage_level,
            neg_voltage_level, timeout, force_adjustment,
            force_adjustment_rel_tol, force_adjustment_pos_voltage_threshold,
            force_adjustment_neg_voltage_threshold,
            force_adjustment_voltage, failure_iteration_threshold,
            time_series_resolution,
            r_off, r_on, d, k_on, k_off, alpha_on, alpha_off, v_on, v_off, x_on,
            x_off, sim_neighbors);
      },
      py::arg("conductance_matrix"), py::arg("device_matrix"),
      py::arg("cuda_malloc_heap_size") = 50, py::arg("rel_tol") = 0.1,
      py::arg("pulse_duration") = 1e-3, py::arg("refactory_period") = 0,
      py::arg("pos_voltage_level") = 1.0, py::arg("neg_voltage_level") = -1.0,
      py::arg("timeout") = 5, py::arg("force_adjustment") = 1e-3,
      py::arg("force_adjustment_rel_tol") = 1e-1,
      py::arg("force_adjustment_pos_voltage_threshold") = 0,
      py::arg("force_adjustment_neg_voltage_threshold") = 0,
      py::arg("force_adjustment_voltage") = 0.2,
      py::arg("failure_iteration_threshold") = 1000,
      py::arg("time_series_resolution") = 1e-10, py::arg("r_off") = 10000,
      py::arg("r_on") = 1000, py::arg("d") = 3e-9, py::arg("k_on") = -10,
      py::arg("k_off") = 5e-4, py::arg("alpha_on") = 3,
      py::arg("alpha_off") = 1, py::arg("v_on") = 0.2, py::arg("v_off") = 0.02,
      py::arg("x_on") = 0, py::arg("x_off") = 3e-9,
      py::arg("simulate_neighbours") = true);

  // Stanford_PKU
  m.def(
      "simulate_passive",
      [&](at::Tensor conductance_matrix, at::Tensor device_matrix,
          int cuda_malloc_heap_size, float rel_tol, float pulse_duration,
          float refactory_period, float pos_voltage_level,
          float neg_voltage_level, float timeout, float force_adjustment,
          float force_adjustment_rel_tol,
          float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold,
          float force_adjustment_voltage,
          int failure_iteration_threshold,
          float time_series_resolution, float r_off, float r_on, float gap_init,
          float g_0, float V_0, float I_0, float read_voltage, float T_init,
          float R_th, float gamma_init, float beta, float t_ox, float F_min,
          float vel_0, float E_a, float a_0, float delta_g_init,
          float model_switch, float T_crit, float T_smth, bool sim_neighbors) {
        return simulate_passive_Stanford_PKU(
            conductance_matrix, device_matrix, cuda_malloc_heap_size, rel_tol,
            pulse_duration, refactory_period, pos_voltage_level,
            neg_voltage_level, timeout, force_adjustment,
            force_adjustment_rel_tol, force_adjustment_pos_voltage_threshold,
            force_adjustment_neg_voltage_threshold,
            force_adjustment_voltage, failure_iteration_threshold,
            time_series_resolution,
            r_off, r_on, gap_init, g_0, V_0, I_0, read_voltage, T_init, R_th,
            gamma_init, beta, t_ox, F_min, vel_0, E_a, a_0, delta_g_init,
            model_switch, T_crit, T_smth, sim_neighbors);
      },
      py::arg("conductance_matrix"), py::arg("device_matrix"),
      py::arg("cuda_malloc_heap_size") = 50, py::arg("rel_tol") = 0.1,
      py::arg("pulse_duration") = 1e-3, py::arg("refactory_period") = 0,
      py::arg("pos_voltage_level") = 1.0, py::arg("neg_voltage_level") = -1.0,
      py::arg("timeout") = 5, py::arg("force_adjustment") = 1e-3,
      py::arg("force_adjustment_rel_tol") = 1e-1,
      py::arg("force_adjustment_pos_voltage_threshold") = 0,
      py::arg("force_adjustment_neg_voltage_threshold") = 0,
      py::arg("force_adjustment_voltage") = 0.2,
      py::arg("failure_iteration_threshold") = 1000,
      py::arg("time_series_resolution") = 1e-10, py::arg("r_off") = 10000,
      py::arg("r_on") = 1000, py::arg("gap_init") = 2e-10,
      py::arg("g_0") = 0.25e-9, py::arg("V_0") = 0.25, py::arg("I_0") = 1000e-6,
      py::arg("read_voltage") = 0.1, py::arg("T_init") = 298,
      py::arg("R_th") = 2.1e3, py::arg("gamma_init") = 16,
      py::arg("beta") = 0.8, py::arg("t_ox") = 12e-9, py::arg("F_min") = 1.4e9,
      py::arg("vel_0") = 10, py::arg("E_a") = 0.6, py::arg("a_0") = 0.25e-9,
      py::arg("delta_g_init") = 0.02, py::arg("model_switch") = 0,
      py::arg("T_crit") = 450, py::arg("T_smth") = 500,
      py::arg("simulate_neighbours") = true);
}