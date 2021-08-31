# include <stdlib.h>

void i4vec2_sorted_uniquely(int n1, int* a1, int *b1, int n2, int *a2, int *b2) {
  int i1 = 0;
  int i2 = 0;
  if (n1 <= 0) {
    return;
  }
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

int i4vec2_sorted_unique_count(int n, int *a1, int *a2) {
  int iu;
  int unique_num = 0;
  if (n <= 0) {
    return unique_num;
  }
  iu = 0;
  unique_num = 1;
  for (int i = 1; i < n; i++) {
    if (a1[i] != a1[iu] || a2[i] != a2[iu]) {
      iu = i;
      unique_num++;
    }
  }
  return unique_num;
}

int i4vec2_compare(int n, int *a1, int *a2, int i, int j) {
  int isgn = 0;
  if (a1[i-1] < a1[j-1]) {
    isgn = -1;
  } else if (a1[i-1] == a1[j-1]) {
    if (a2[i-1] < a2[j-1]) {
      isgn = -1;
    } else if (a2[i-1] < a2[j-1]) {
      isgn = 0;
    } else if (a2[j-1] < a2[i-1]) {
      isgn = +1;
    }
  } else if (a1[j-1] < a1[i-1]) {
    isgn = +1;
  }
  return isgn;
}

void sort_heap_external(int n, int *indx, int *i, int *j, int isgn) {
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

void i4vec2_sort_a(int n, int* a1, int* a2) {
  int i = 0;
  int indx = 0;
  int isgn = 0;
  int j = 0;
  int temp;
  // Call the external heap sorter.
  for (;;) {
    sort_heap_external(n, &indx, &i, &j, isgn);
  // Interchange the I and J objects.
    if (0 < indx) {
      temp = a1[i-1];
      a1[i-1] = a1[j-1];
      a1[j-1] = temp;
      temp = a2[i-1];
      a2[i-1] = a2[j-1];
      a2[j-1] = temp;
    } else if (indx < 0) {
      // Compare the I and J objects.
      isgn = i4vec2_compare(n, a1, a2, i, j);
    } else if (indx == 0) {
      break;
    }
  }
  return;
}

int *i4vec_copy_new(int n, int* a1) {
  int *a2 = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    a2[i] = a1[i];
  }
  return a2;
}

int st_to_cc_size(int nst, int* ist, int* jst) {
  int *ist2;
  int *jst2;
  int ncc;
  // Make copies so the sorting doesn't confuse the user.
  ist2 = i4vec_copy_new(nst, ist);
  jst2 = i4vec_copy_new(nst, jst);
  // Sort by column first, then row.
  i4vec2_sort_a(nst, jst2, ist2);
  // Count the unique pairs.
  ncc = i4vec2_sorted_unique_count(nst, jst2, ist2);
  free(ist2);
  free(jst2);
  return ncc;
}

void st_to_cc_index(int nst, int* ist, int *jst, int ncc, int n, int *icc, int *ccc) {
  int *ist2;
  int j;
  int *jcc;
  int jhi;
  int jlo;
  int *jst2;
  // Make copies so the sorting doesn't confuse the user.
  ist2 = i4vec_copy_new(nst, ist);
  jst2 = i4vec_copy_new(nst, jst);
  // Sort the elements.
  i4vec2_sort_a(nst, jst2, ist2);
  // Get the unique elements.
  jcc = (int *)malloc(ncc * sizeof(int));
  i4vec2_sorted_uniquely(nst, jst2, ist2, ncc, jcc, icc);
  // Compress the column index.
  ccc[0] = 0;
  jlo = 0;
  for (int k = 0; k < ncc; k++) {
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
  free(ist2);
  free(jcc);
  free(jst2);
  return;
}

template <class T> T *st_to_cc_values (int nst, int *ist, int *jst, T *ast, int ncc, int n, int* icc, int *ccc) {
  T *acc;
  int chi;
  int clo;
  int fail;
  int i;
  int j;
  int kcc;
  acc = (T *)malloc(ncc * sizeof(T));
  for (i = 0; i < ncc; i++) {
    acc[i] = (T)0.0;
  }
  for (int kst = 0; kst < nst; kst++) {
    i = ist[kst];
    j = jst[kst];
    clo = ccc[j];
    chi = ccc[j+1];
    fail = 1;
    for (kcc = clo; kcc < chi; kcc++) {
      if (icc[kcc] == i) {
        acc[kcc] = acc[kcc] + ast[kst];
        fail = 0;
        break;
      }
    }
    if ( fail )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "ST_TO_CC_VALUES - Fatal error!\n" );
      fprintf ( stderr, "  ST entry cannot be located in CC array.\n" );
      fprintf ( stderr, "  ST index KST    = %d\n", kst );
      fprintf ( stderr, "  ST row IST(KST) = %d\n", ist[kst] );
      fprintf ( stderr, "  ST col JST(KST) = %d\n", jst[kst] );
      fprintf ( stderr, "  ST val AST(KST) = %g\n", ast[kst] );
      exit ( 1 );
    }
  }
  return acc;
}