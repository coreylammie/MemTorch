void gen_tiles_bindings_gpu(py::module_ &m);
std::tuple<at::Tensor, at::Tensor>
gen_tiles(at::Tensor tensor, int tile_shape[2], bool input,
          torch::TensorOptions tensor_options);