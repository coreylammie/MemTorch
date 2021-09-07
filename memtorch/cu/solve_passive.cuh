__device__ Eigen::VectorXf solve_passive(Eigen::MatrixXf conductance_matrix,
                                         int m, int n, Eigen::VectorXf V_WL,
                                         float R_source, float R_line) {
  int non_zero_elements = 8 * m * n - 2 * m - 2 * n;
  printf("%d\n", non_zero_elements * sizeof(double) +
                     2 * non_zero_elements * sizeof(csi) +
                     2 * m * n * sizeof(double));
  csi *ABCD_matrix_indices_x =
      (csi *)cs_malloc(sizeof(csi) * non_zero_elements);
  csi *ABCD_matrix_indices_y =
      (csi *)cs_malloc(sizeof(csi) * non_zero_elements);
  double *ABCD_matrix_values =
      (double *)cs_malloc(sizeof(double) * non_zero_elements);
  double *E_matrix = (double *)cs_malloc(sizeof(double) * 2 * m * n);
  // A, B, and E (partial) matrices
  int index = 0;
  for (int i = 0; i < m; i++) {
    // A matrix
    ABCD_matrix_indices_x[index] = (csi)(i * n);
    ABCD_matrix_indices_y[index] = (csi)(i * n);
    ABCD_matrix_values[index] = (double)conductance_matrix(0, 0) +
                                1.0 / (double)R_source + 1.0 / (double)R_line;
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
      index++;
    }
  }
  // free(V_WL.data());
  // free(&V_WL);
  // free(V_BL.data());
  // free(&V_BL);
  V_WL.resize(0, 0);
  // Solve (ABCD)V = E
  cs *ABCD_matrix = (cs *)cs_malloc(sizeof(cs));
  ABCD_matrix->m = (csi)(2 * m * n);
  ABCD_matrix->n = (csi)(2 * m * n);
  ABCD_matrix->nzmax = (csi)non_zero_elements;
  ABCD_matrix->nz = (csi)non_zero_elements;
  swap(ABCD_matrix->i, ABCD_matrix_indices_x);
  swap(ABCD_matrix->p, ABCD_matrix_indices_y);
  swap(ABCD_matrix->x, ABCD_matrix_values);
  free(ABCD_matrix_indices_x);
  free(ABCD_matrix_indices_y);
  free(ABCD_matrix_values);
  cs *ABCD_matrix_compressed = cs_compress(ABCD_matrix);
  cs_spfree(ABCD_matrix);
  printf("Solve_D\n");
  cs_qrsol(1, ABCD_matrix_compressed, E_matrix);
  cs_spfree(ABCD_matrix_compressed);
  printf("Solve_E\n");
  Eigen::MatrixXf I_applied_tensor = Eigen::MatrixXf::Zero(m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      I_applied_tensor(i, j) =
          (E_matrix[n * i + j] - E_matrix[m * n + n * i + j]) *
          conductance_matrix(i, j);
    }
  }
  // free(&conductance_matrix.data);
  // free(&conductance_matrix);
  conductance_matrix.resize(0, 0);
  free(E_matrix);
  return I_applied_tensor.colwise().sum();
}
