#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER inline __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER inline
#endif

CUDA_CALLABLE_MEMBER int i4vec2_sorted_unique_count(int n, int a1[], int a2[]) {
  int i;
  int iu = 0;
  int unique_num = 0;
  if (n <= 0) {
    return unique_num;
  }
  unique_num = 1;
  for (i = 1; i < n; i++) {
    if (a1[i] != a1[iu] || a2[i] != a2[iu]) {
      iu = i;
      unique_num = unique_num + 1;
    }
  }
  return unique_num;
}

CUDA_CALLABLE_MEMBER int i4vec2_compare(int n, int *a1, int *a2, int i, int j) {
  int isgn = 0;
  if (a1[i - 1] < a1[j - 1]) {
    isgn = -1;
  } else if (a1[i - 1] == a1[j - 1]) {
    if (a2[i - 1] < a2[j - 1]) {
      isgn = -1;
    } else if (a2[i - 1] < a2[j - 1]) {
      isgn = 0;
    } else if (a2[j - 1] < a2[i - 1]) {
      isgn = +1;
    }
  } else if (a1[j - 1] < a1[i - 1]) {
    isgn = +1;
  }
  return isgn;
}

CUDA_CALLABLE_MEMBER void sort_heap_external(int n, int *indx, int *i, int *j,
                                             int isgn) {
  static int i_save = 0;
  static int j_save = 0;
  static int k = 0;
  static int k1 = 0;
  static int n1 = 0;
  if (*indx == 0) {
    i_save = 0;
    j_save = 0;
    k = n / 2;
    k1 = k;
    n1 = n;
  } else if (*indx < 0) {
    if (*indx == -2) {
      if (isgn < 0) {
        i_save = i_save + 1;
      }
      j_save = k1;
      k1 = i_save;
      *indx = -1;
      *i = i_save;
      *j = j_save;
      return;
    }
    if (0 < isgn) {
      *indx = 2;
      *i = i_save;
      *j = j_save;
      return;
    }
    if (k <= 1) {
      if (n1 == 1) {
        i_save = 0;
        j_save = 0;
        *indx = 0;
      } else {
        i_save = n1;
        j_save = 1;
        n1 = n1 - 1;
        *indx = 1;
      }
      *i = i_save;
      *j = j_save;
      return;
    }
    k = k - 1;
    k1 = k;
  } else if (*indx == 1) {
    k1 = k;
  }
  for (;;) {
    i_save = 2 * k1;
    if (i_save == n1) {
      j_save = k1;
      k1 = i_save;
      *indx = -1;
      *i = i_save;
      *j = j_save;
      return;
    } else if (i_save <= n1) {
      j_save = i_save + 1;
      *indx = -2;
      *i = i_save;
      *j = j_save;
      return;
    }
    if (k <= 1) {
      break;
    }
    k = k - 1;
    k1 = k;
  }
  if (n1 == 1) {
    i_save = 0;
    j_save = 0;
    *indx = 0;
    *i = i_save;
    *j = j_save;
  } else {
    i_save = n1;
    j_save = 1;
    n1 = n1 - 1;
    *indx = 1;
    *i = i_save;
    *j = j_save;
  }
  return;
}

CUDA_CALLABLE_MEMBER void i4vec2_sort_a(int n, int *a1, int *a2) {
  int i = 0;
  int j = 0;
  int indx = 0;
  int isgn = 0;
  int temp;
  for (;;) {
    sort_heap_external(n, &indx, &i, &j, isgn);
    if (0 < indx) {
      temp = a1[i - 1];
      a1[i - 1] = a1[j - 1];
      a1[j - 1] = temp;

      temp = a2[i - 1];
      a2[i - 1] = a2[j - 1];
      a2[j - 1] = temp;
    } else if (indx < 0) {
      isgn = i4vec2_compare(n, a1, a2, i, j);
    } else if (indx == 0) {
      break;
    }
  }
  return;
}

void i4vec2_sorted_uniquely(int n1, int *a1, int *b1, int n2, int *a2,
                            int *b2) {
  int i1 = 0;
  int i2 = 0;
  if (n1 <= 0) {
    return;
  }
  // printf("B");
  a2[i2] = a1[i1];
  b2[i2] = b1[i1];
  for (i1 = 1; i1 < n1; i1++) {
    if (a1[i1] != a2[i2] || b1[i1] != b2[i2]) {
      i2 = i2 + 1;
      a2[i2] = a1[i1];
      b2[i2] = b1[i1];
    }
  }
  return;
}

CUDA_CALLABLE_MEMBER int st_to_cc_size(int nst, int ist[], int jst[]) {
  int ncc;
  i4vec2_sort_a(nst, jst, ist);
  ncc = i4vec2_sorted_unique_count(nst, jst, ist);
  return ncc;
}

CUDA_CALLABLE_MEMBER int st_to_cc_index(int nst, int *ist, int *jst, int ncc,
                                        int n, int *icc, int *ccc, int *jcc) {
  int j;
  int jhi;
  int jlo;
  int k;
  i4vec2_sort_a(nst, jst, ist);
  // jcc = (int *)malloc(ncc * sizeof(int));
  // i4vec2_sorted_uniquely(nst, jst, ist, ncc, jcc, icc);
  int i1 = 0;
  int i2 = 0;
  jcc[i2] = jst[i1];
  icc[i2] = ist[i1];
  for (i1 = 1; i1 < nst; i1++) {
    if (jst[i1] != jcc[i2] || ist[i1] != icc[i2]) {
      i2 = i2 + 1;
      jcc[i2] = jst[i1];
      icc[i2] = ist[i1];
    }
  }
  ccc[0] = 0;
  jlo = 0;
  for (k = 0; k < ncc; k++) {
    jhi = jcc[k];
    if (jhi != jlo) {
      for (j = jlo + 1; j <= jhi; j++) {
        ccc[j] = k;
      }
      jlo = jhi;
    }
  }
  jhi = n;
  for (j = jlo + 1; j <= jhi; j++) {
    ccc[j] = ncc;
  }
  return ncc;
}

CUDA_CALLABLE_MEMBER void st_to_cc_values(int nst, int *ist, int *jst,
                                          double *ast, int ncc, int n, int *icc,
                                          int *ccc, double *acc) {
  int chi;
  int clo;
  int fail;
  int i;
  int j;
  int kcc;
  int kst;
  for (i = 0; i < nst; i++) {
    acc[i] = 0.0;
  }
  printf("%d/%d\n", nst, ncc);
  for (kst = 0; kst < nst; kst++) {
    i = ist[kst];
    j = jst[kst];
    clo = ccc[j];
    chi = ccc[j + 1];
    fail = 1;
    for (kcc = clo; kcc < chi; kcc++) {
      if (icc[kcc] == i) {
        acc[kcc] = acc[kcc] + ast[kst];
        fail = 0;
        break;
      }
    }
    if (fail) {
      printf("%d\n", i);
      printf("%d\n", j);
      printf("  ST entry cannot be located in CC array.\n");
      printf("  ST index KST    = %d\n", kst);
      printf("  ST row IST(KST) = %d\n", ist[kst]);
      printf("  ST col JST(KST) = %d\n", jst[kst]);
      printf("  ST val AST(KST) = %g\n", ast[kst]);
      exit(1);
    }
  }
  return;
}

/* p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c */
CUDA_CALLABLE_MEMBER double cs_cumsum(int *p, int *c, int n) {
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

/* C = compressed-column form of a triplet matrix T */
CUDA_CALLABLE_MEMBER void cs_compress(int nst, int m, int n, int *ist, int *jst,
                                      double *ast, int *ist_compressed,
                                      int *jst_compressed,
                                      double *ast_compressed) {
  int p;
  int *w = (int *)malloc(sizeof(int) * n);
  for (int i = 0; i < n; i++) {
    w[i] = 0;
  }
  for (int k = 0; k < nst; k++) {
    w[jst[k]]++;
  }                                // column counts
  cs_cumsum(jst_compressed, w, n); // column pointers
  for (int k = 0; k < nst; k++) {
    p = w[jst[k]]++;
    // printf("%d\n", p);
    ist_compressed[p] = ist[k]; // A(i,j) is the pth entry in C
    ast_compressed[p] = ast[k];
  }
  free(w);
  return;
}