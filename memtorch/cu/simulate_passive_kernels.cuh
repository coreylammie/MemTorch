at::Tensor simulate_passive_dd(at::Tensor conductance_matrix, at::Tensor device_matrix, int cuda_malloc_heap_size, float rel_tol,
          float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
          float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold, float time_series_resolution , float r_off, float r_on, float A_p, float A_n, float t_p, float t_n,
          float k_p, float k_n, std::vector<float> r_p, std::vector<float> r_n, float a_p, float a_n, float b_p, float b_n, bool sim_neighbors);

at::Tensor simulate_passive_linearIonDrift(at::Tensor conductance_matrix, at::Tensor device_matrix, int cuda_malloc_heap_size, float rel_tol,
          float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
          float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold, float time_series_resolution , float r_off, float r_on, float u_v,
          float d,float pos_write_threshold, float neg_write_threshold, float p, bool sim_neighbors);

at::Tensor simulate_passive_Stanford_PKU(at::Tensor conductance_matrix, at::Tensor device_matrix, int cuda_malloc_heap_size, float rel_tol,
          float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
          float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold, float time_series_resolution , float r_off, float r_on, float gap_init,
          float g_0, float V_0, float I_0, float read_voltage, float T_init, float R_th, float gamma_init,
          float beta, float t_ox, float F_min, float vel_0, float E_a, float a_0, float delta_g_init,
          float model_switch, float T_crit, float T_smth, bool sim_neighbors);

at::Tensor simulate_passive_VTEAM(at::Tensor conductance_matrix, at::Tensor device_matrix, int cuda_malloc_heap_size, float rel_tol,
          float pulse_duration, float refactory_period, float pos_voltage_level, float neg_voltage_level,
          float timeout, float force_adjustment, float force_adjustment_rel_tol, float force_adjustment_pos_voltage_threshold,
          float force_adjustment_neg_voltage_threshold, float time_series_resolution , float r_off, float r_on, float d,
          float k_on, float k_off, float alpha_on,  float alpha_off, float v_on, float v_off, float x_on, float x_off, bool sim_neighbors);


int countOccurrences(int arr[], int n, int x);