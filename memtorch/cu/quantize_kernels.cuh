at::Tensor quantize(at::Tensor tensor, int bits, float overflow_rate,
                    int quant_method, float min, float max);