void quantize_bindings(py::module_ &m);

template <class T>
void quantize_element(T *tensor, int index, T *quant_levels,
                      int num_quant_levels) {
  int middle_point;         // Middle point
  int optimal_point = 0;    // Optimal point
  int l = 0;                // Lower bound
  int h = num_quant_levels; // Higher bound
  T difference =
      std::numeric_limits<T>().max(); // Difference between a given point
                                      // and the current middle point
  while (l <= h) {
    middle_point = l + (h - l) / 2;
    if (fabs(tensor[index] - quant_levels[middle_point]) < difference) {
      difference = abs(tensor[index] - quant_levels[middle_point]);
      optimal_point = middle_point;
    }
    if (quant_levels[middle_point] < tensor[index]) {
      l = middle_point + 1;
    } else {
      h = middle_point - 1;
    }
  }
  tensor[index] = quant_levels[optimal_point];
}

template <class T>
T det_integral(at::Tensor tensor, T overflow_rate, T min, T max) {
  if (overflow_rate > 1.0) {
    throw std::invalid_argument("Invalid overflow_rate value.");
  } else {
    tensor = std::get<0>(at::sort(at::flatten(at::abs(tensor)), -1, true));
    int64_t tensor_numel = tensor.numel();
    if ((min != NULL) || (max != NULL)) {
      T max_bound;
      if ((min != NULL) && (max != NULL)) {
        max_bound = std::max(std::abs(min), std::abs(max));
      } else if (min != NULL) {
        max_bound = std::abs(min);
      } else if (max != NULL) {
        max_bound = std::abs(max);
      }
      if (max_bound > tensor[0].item<T>()) {
        tensor[0] = max_bound;
      }
    }
    T *data_ptr = tensor.data_ptr<T>();
    return ceil(
        log2(data_ptr[std::min<int>((int)round(overflow_rate * tensor_numel),
                                    tensor_numel - 1)] +
             1e-12f));
  }
}

template <class T>
T det_sf(at::Tensor tensor, int bits, T overflow_rate, T min, T max) {
  return 1 - bits + det_integral<T>(tensor, overflow_rate, min, max);
}

template <class T>
at::Tensor linear_quantize(at::Tensor tensor, T sf, int bits, T overflow_rate) {
  T delta = pow((T)2.0, sf);
  T bound = pow((T)2.0, bits - 1);
  return at::clamp(at::floor(tensor / pow((T)2.0, sf) + 0.5), -bound,
                   bound - 1) *
         delta;
}

template <class T> void set_average(at::Tensor tensor, T *input_tensor_ptr) {
  T mean_value = at::flatten(tensor).item<T>();
#pragma omp parallel for
  for (int i = 0; i < tensor.numel(); i++) {
    input_tensor_ptr[i] = mean_value;
  }
}

template <class T> void parse_min_max(T *min, T *max) {
  if (isnan(*min)) {
    *min = NULL;
  }
  if (isnan(*max)) {
    *max = NULL;
  }
}

template <class T>
void quantize(at::Tensor tensor, int n_quant_levels, T min = NULL,
              T max = NULL) {
  parse_min_max<T>(&min, &max);
  T *input_tensor_ptr = tensor.data_ptr<T>();
  if (n_quant_levels == 1) {
    set_average<T>(tensor, input_tensor_ptr);
    return;
  }
  if (min == NULL) {
    min = at::flatten(tensor).min().item<T>();
  }
  if (max == NULL) {
    max = at::flatten(tensor).max().item<T>();
  }
  at::TensorOptions options;
  if (typeid(T) == typeid(float)) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
  } else {
    options = torch::TensorOptions().dtype(torch::kFloat64);
  }
  at::Tensor quant_levels = at::linspace(min, max, n_quant_levels, options);
#pragma omp parallel for
  for (int i = 0; i < tensor.numel(); i++) {
    quantize_element<T>(input_tensor_ptr, i, quant_levels.data_ptr<T>(),
                        n_quant_levels);
  }
  return;
}

template <class T>
void quantize(at::Tensor tensor, int n_quant_levels, at::Tensor min,
              at::Tensor max) {
  T *input_tensor_ptr = tensor.data_ptr<T>();
  if (n_quant_levels == 1) {
    set_average<T>(tensor, input_tensor_ptr);
    return;
  }
  T *min_ptr = min.data_ptr<T>();
  T *max_ptr = max.data_ptr<T>();
  at::TensorOptions options;
  if (typeid(T) == typeid(float)) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
  } else {
    options = torch::TensorOptions().dtype(torch::kFloat64);
  }
#pragma omp parallel for
  for (int i = 0; i < tensor.numel(); i++) {
    torch::Tensor quant_levels =
                      at::linspace(min_ptr[i], max_ptr[i], n_quant_levels),
                  options;
    quantize_element<T>(input_tensor_ptr, i, quant_levels.data_ptr<T>(),
                        n_quant_levels);
  }
}

template <class T>
void quantize(at::Tensor tensor, int bits, T overflow_rate,
              int quant_method = 0, T min = NULL, T max = NULL) {
  parse_min_max(&min, &max);
  T *input_tensor_ptr = tensor.data_ptr<T>();
  T *quantized_tensor_ptr = nullptr;
  if ((int)at::numel(std::get<0>(at::unique_consecutive(tensor))) == 1) {
    return;
  } else {
    if (bits == 1) {
      set_average<T>(tensor, input_tensor_ptr);
      return;
    } else {
      if (min != NULL) {
        tensor = at::clamp_min(tensor, min);
      }
      if (max != NULL) {
        tensor = at::clamp_max(tensor, max);
      }
      if ((quant_method == 0) || (quant_method == 1)) {
        if (quant_method == 0) {
          // linear
          at::Tensor quantized_tensor = linear_quantize<T>(
              tensor, det_sf<T>(tensor, bits, overflow_rate, min, max), bits,
              overflow_rate);
          T *quantized_tensor_ptr = quantized_tensor.data_ptr<T>();
#pragma omp parallel for
          for (int i = 0; i < tensor.numel(); i++) {
            input_tensor_ptr[i] = quantized_tensor_ptr[i];
          }
        } else {
          // log
          at::Tensor s = at::sign(tensor);
          T sf = det_sf<T>(tensor, bits, overflow_rate, min, max);
          tensor = at::log(at::abs(tensor)).clamp_min_(1e-20f);
          at::Tensor quantized_tensor =
              at::exp(linear_quantize<T>(tensor, sf, bits - 1, overflow_rate)) *
              s;
          T *quantized_tensor_ptr = quantized_tensor.data_ptr<T>();
#pragma omp parallel for
          for (int i = 0; i < tensor.numel(); i++) {
            input_tensor_ptr[i] = quantized_tensor_ptr[i];
          }
        }
      } else {
        throw std::invalid_argument(
            "Invalid quant_method: 0 -> linear, 1 -> log.");
      }
    }
  }
}