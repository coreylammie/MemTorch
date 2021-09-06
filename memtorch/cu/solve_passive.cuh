__device__ Eigen::VectorXf solve_passive(Eigen::MatrixXf conductance_matrix,
                                         int m, int n, Eigen::VectorXf V_WL,
                                         Eigen::VectorXf V_BL, float R_source,
                                         float R_line) {
  int non_zero_elements = 8 * m * n - 2 * m - 2 * n;
  csi *ABCD_matrix_indices_x = (csi *)malloc(sizeof(csi) * non_zero_elements);
  csi *ABCD_matrix_indices_y = (csi *)malloc(sizeof(csi) * non_zero_elements);
  double *ABCD_matrix_values =
      (double *)malloc(sizeof(double) * non_zero_elements);
  double *E_matrix = (double *)malloc(sizeof(double) * 2 * m * n);
  // A, B, and E (partial) matrices
  int index = 0;
  for (int i = 0; i < m; i++) {
    // A matrix
    ABCD_matrix_indices_x[index] = (csi)(i * n);
    ABCD_matrix_indices_y[index] = (csi)(i * n);
    ABCD_matrix_values[index] = (double)conductance_matrix(i, 0) +
                                1.0 / (double)R_source + 1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = (csi)(i * n + 1);
    ABCD_matrix_indices_y[index] = (csi)(i * n);
    ABCD_matrix_values[index] = -1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = (csi)(i * n);
    ABCD_matrix_indices_y[index] = (csi)(i * n + 1);
    ABCD_matrix_values[index] = -1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = (csi)(i * n + (n - 1));
    ABCD_matrix_indices_y[index] = (csi)(i * n + (n - 1));
    ABCD_matrix_values[index] =
        (double)conductance_matrix(i, n - 1) + 1.0 / (double)R_line;
    index++;
    // B matrix
    ABCD_matrix_indices_x[index] = (csi)(i * n);
    ABCD_matrix_indices_y[index] = (csi)(i * n + (m * n));
    ABCD_matrix_values[index] = (double)-conductance_matrix(i, 0);
    index++;
    ABCD_matrix_indices_x[index] = (csi)(i * n + (n - 1));
    ABCD_matrix_indices_y[index] = (csi)(i * n + (n - 1) + (m * n));
    ABCD_matrix_values[index] = (double)-conductance_matrix(i, n - 1);
    index++;
    // E matrix
    E_matrix[i * n] = (double)V_WL[i] / (double)R_source;
    for (int j = 1; j < n - 1; j++) {
      // A matrix
      ABCD_matrix_indices_x[index] = (csi)(i * n + j);
      ABCD_matrix_indices_y[index] = (csi)(i * n + j);
      ABCD_matrix_values[index] =
          (double)conductance_matrix(i, j) + 2.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = (csi)(i * n + j + 1);
      ABCD_matrix_indices_y[index] = (csi)(i * n + j);
      ABCD_matrix_values[index] = -1.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = (csi)(i * n + j);
      ABCD_matrix_indices_y[index] = (csi)(i * n + j + 1);
      ABCD_matrix_values[index] = -1.0 / (double)R_line;
      index++;
      // B matrix
      ABCD_matrix_indices_x[index] = (csi)(i * n + j);
      ABCD_matrix_indices_y[index] = (csi)(i * n + j + (m * n));
      ABCD_matrix_values[index] = (double)-conductance_matrix(i, j);
      index++;
    }
  }
  // C, D, and E (partial) matrices
  for (int j = 0; j < n; j++) {
    // D matrix
    ABCD_matrix_indices_x[index] = (csi)(m * n + (j * m));
    ABCD_matrix_indices_y[index] = (csi)(m * n + j);
    ABCD_matrix_values[index] =
        -1.0 / (double)R_line - conductance_matrix(0, j);
    index++;
    ABCD_matrix_indices_x[index] = (csi)(m * n + (j * m));
    ABCD_matrix_indices_y[index] = (csi)(m * n + j + n);
    ABCD_matrix_values[index] = 1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = (csi)(m * n + (j * m) + m - 1);
    ABCD_matrix_indices_y[index] = (csi)(m * n + (n * (m - 2)) + j);
    ABCD_matrix_values[index] = 1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = (csi)(m * n + (j * m) + m - 1);
    ABCD_matrix_indices_y[index] = (csi)(m * n + (n * (m - 1)) + j);
    ABCD_matrix_values[index] = -1.0 / (double)R_source -
                                conductance_matrix(m - 1, j) -
                                1.0 / (double)R_line;
    index++;
    // C matrix
    ABCD_matrix_indices_x[index] = (csi)(j * m + (m * n));
    ABCD_matrix_indices_y[index] = (csi)j;
    ABCD_matrix_values[index] = (double)conductance_matrix(0, j);
    index++;
    ABCD_matrix_indices_x[index] = (csi)(j * m + (m - 1) + (m * n));
    ABCD_matrix_indices_y[index] = (csi)(n * (m - 1) + j);
    ABCD_matrix_values[index] = (double)conductance_matrix(m - 1, j);
    index++;
    // E matrix
    E_matrix[m * n + (j + 1) * m - 1] = -V_BL(j) / (double)R_source;
    for (int i = 1; i < m - 1; i++) {
      // D matrix
      ABCD_matrix_indices_x[index] = (csi)(m * n + (j * m) + i);
      ABCD_matrix_indices_y[index] = (csi)(m * n + (n * (i - 1)) + j);
      ABCD_matrix_values[index] = 1.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = (csi)(m * n + (j * m) + i);
      ABCD_matrix_indices_y[index] = (csi)(m * n + (n * (i + 1)) + j);
      ABCD_matrix_values[index] = 1.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = (csi)(m * n + (j * m) + i);
      ABCD_matrix_indices_y[index] = (csi)(m * n + (n * i) + j);
      ABCD_matrix_values[index] =
          (double)-conductance_matrix(i, j) - 2.0 / (double)R_line;
      index++;
      // C matrix
      ABCD_matrix_indices_x[index] = (csi)(j * m + i + (m * n));
      ABCD_matrix_indices_y[index] = (csi)(n * i + j);
      ABCD_matrix_values[index] = (double)conductance_matrix(i, j);
    }
  }
  // Solve (ABCD)V = E
  cs *ABCD_matrix = (cs *)malloc(sizeof(cs));
  ABCD_matrix->m = (csi)(2 * m * n);
  ABCD_matrix->n = (csi)(2 * m * n);
  ABCD_matrix->nzmax = (csi)non_zero_elements;
  ABCD_matrix->nz = (csi)non_zero_elements;
  swap(ABCD_matrix->i, ABCD_matrix_indices_x);
  swap(ABCD_matrix->p, ABCD_matrix_indices_y);
  swap(ABCD_matrix->x, ABCD_matrix_values);
  cs *ABCD_matrix_compressed = cs_compress(ABCD_matrix);
  cs_spfree(ABCD_matrix);
  cs_qrsol(1, ABCD_matrix_compressed, E_matrix);
  Eigen::MatrixXf V_applied_tensor = Eigen::MatrixXf::Zero(m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      V_applied_tensor(i, j) =
          E_matrix[n * i + j] - E_matrix[m * n + n * i + j];
    }
  }
  return V_applied_tensor.cwiseProduct(conductance_matrix); //.colwise.sum();
  // return at::sum(at::mul(V_applied_tensor, conductance_matrix), 0);
}
