void quantize_bindings(py::module_ &m);
void quantize(at::Tensor tensor, int bits, float overflow_rate,
              int quant_method = 0, float min = NULL, float max = NULL);