at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
    at::Tensor V_BL, int ADC_resolution, 
    float overflow_rate, int quant_method,
    float R_source, float R_line,
    bool det_readout_currents);