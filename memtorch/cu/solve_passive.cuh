__device__ double cumsum(int *p, int *c, int n) {
  int nz = 0;
  double nz2 = 0;
  for (int i = 0; i < n; i++) {
    p[i] = nz;
    nz += c[i];
    nz2 += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz2;
}

__device__ void cs_compress(int nst, int m, int n, int *ist, int *jst,
                            double *ast, int *ist_compressed,
                            int *jst_compressed, double *ast_compressed) {
  int p;
  int *w = (int *)malloc(sizeof(int) * n);
  for (int i = 0; i < n; i++) {
    w[i] = 0;
  }
  for (int k = 0; k < nst; k++) {
    w[jst[k]]++;
  }
  cumsum(jst_compressed, w, n);
  for (int k = 0; k < nst; k++) {
    p = w[jst[k]]++;
    ist_compressed[p] = ist[k];
    ast_compressed[p] = ast[k];
  }
  free(w);
  return;
}

__device__ void
construct_ABCD_E(Eigen::MatrixXf conductance_matrix, int m, int n,
                 Eigen::VectorXf V_WL, float R_source, float R_line,
                 int *ABCD_matrix_indices_x, int *ABCD_matrix_indices_y,
                 double *ABCD_matrix_values, int *ABCD_matrix_compressed_rows,
                 int *ABCD_matrix_compressed_columns,
                 double *ABCD_matirx_compressed_values, double *E_matrix) {
  // A, B, and E (partial) matrices
  int nonzero_elements = 8 * m * n - 2 * m - 2 * n;
  int index = 0;
  for (int i = 0; i < m; i++) {
    // A matrix
    ABCD_matrix_indices_x[index] = i * n;
    ABCD_matrix_indices_y[index] = i * n;
    ABCD_matrix_values[index] = (double)conductance_matrix(i, 0) +
                                1.0 / (double)R_source + 1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = i * n + 1;
    ABCD_matrix_indices_y[index] = i * n;
    ABCD_matrix_values[index] = -1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = i * n;
    ABCD_matrix_indices_y[index] = i * n + 1;
    ABCD_matrix_values[index] = -1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = i * n + (n - 1);
    ABCD_matrix_indices_y[index] = i * n + (n - 1);
    ABCD_matrix_values[index] =
        (double)conductance_matrix(i, n - 1) + 1.0 / (double)R_line;
    index++;
    // B matrix
    ABCD_matrix_indices_x[index] = i * n;
    ABCD_matrix_indices_y[index] = i * n + (m * n);
    ABCD_matrix_values[index] = (double)-conductance_matrix(i, 0);
    index++;
    ABCD_matrix_indices_x[index] = i * n + (n - 1);
    ABCD_matrix_indices_y[index] = i * n + (n - 1) + (m * n);
    ABCD_matrix_values[index] = (double)-conductance_matrix(i, n - 1);
    index++;
    // E matrix
    E_matrix[i * n] = (double)V_WL[i] / (double)R_source;
    for (int j = 1; j < n - 1; j++) {
      // A matrix
      ABCD_matrix_indices_x[index] = i * n + j;
      ABCD_matrix_indices_y[index] = i * n + j;
      ABCD_matrix_values[index] =
          (double)conductance_matrix(i, j) + 2.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = i * n + j + 1;
      ABCD_matrix_indices_y[index] = i * n + j;
      ABCD_matrix_values[index] = -1.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = i * n + j;
      ABCD_matrix_indices_y[index] = i * n + j + 1;
      ABCD_matrix_values[index] = -1.0 / (double)R_line;
      index++;
      // B matrix
      ABCD_matrix_indices_x[index] = i * n + j;
      ABCD_matrix_indices_y[index] = i * n + j + (m * n);
      ABCD_matrix_values[index] = (double)-conductance_matrix(i, j);
      index++;
    }
  }
  // C, D, and E (partial) matrices
  for (int j = 0; j < n; j++) {
    // D matrix
    ABCD_matrix_indices_x[index] = m * n + (j * m);
    ABCD_matrix_indices_y[index] = m * n + j;
    ABCD_matrix_values[index] =
        -1.0 / (double)R_line - conductance_matrix(0, j);
    index++;
    ABCD_matrix_indices_x[index] = m * n + (j * m);
    ABCD_matrix_indices_y[index] = m * n + j + n;
    ABCD_matrix_values[index] = 1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = m * n + (j * m) + m - 1;
    ABCD_matrix_indices_y[index] = m * n + (n * (m - 2)) + j;
    ABCD_matrix_values[index] = 1.0 / (double)R_line;
    index++;
    ABCD_matrix_indices_x[index] = m * n + (j * m) + m - 1;
    ABCD_matrix_indices_y[index] = m * n + (n * (m - 1)) + j;
    ABCD_matrix_values[index] = -1.0 / (double)R_source -
                                conductance_matrix(m - 1, j) -
                                1.0 / (double)R_line;
    index++;
    // C matrix
    ABCD_matrix_indices_x[index] = j * m + (m * n);
    ABCD_matrix_indices_y[index] = j;
    ABCD_matrix_values[index] = (double)conductance_matrix(0, j);
    index++;
    ABCD_matrix_indices_x[index] = j * m + (m - 1) + (m * n);
    ABCD_matrix_indices_y[index] = n * (m - 1) + j;
    ABCD_matrix_values[index] = (double)conductance_matrix(m - 1, j);
    index++;
    for (int i = 1; i < m - 1; i++) {
      // D matrix
      ABCD_matrix_indices_x[index] = m * n + (j * m) + i;
      ABCD_matrix_indices_y[index] = m * n + (n * (i - 1)) + j;
      ABCD_matrix_values[index] = 1.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = m * n + (j * m) + i;
      ABCD_matrix_indices_y[index] = m * n + (n * (i + 1)) + j;
      ABCD_matrix_values[index] = 1.0 / (double)R_line;
      index++;
      ABCD_matrix_indices_x[index] = m * n + (j * m) + i;
      ABCD_matrix_indices_y[index] = m * n + (n * i) + j;
      ABCD_matrix_values[index] =
          (double)-conductance_matrix(i, j) - 2.0 / (double)R_line;
      index++;
      // C matrix
      ABCD_matrix_indices_x[index] = j * m + i + (m * n);
      ABCD_matrix_indices_y[index] = n * i + j;
      ABCD_matrix_values[index] = (double)conductance_matrix(i, j);
      index++;
    }
  }
  V_WL.resize(0, 0);
  cs_compress(nonzero_elements, 2 * n * m, 2 * n * m, ABCD_matrix_indices_x,
              ABCD_matrix_indices_y, ABCD_matrix_values,
              ABCD_matrix_compressed_rows, ABCD_matrix_compressed_columns,
              ABCD_matirx_compressed_values);
}