at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2],
                       int ADC_resolution, float overflow_rate,
                       int quant_method, int cuda_malloc_heap_size);