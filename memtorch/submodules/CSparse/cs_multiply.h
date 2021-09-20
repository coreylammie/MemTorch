/* C = A*B */
CUDA_CALLABLE_MEMBER
cs *cs_multiply(const cs *A, const cs *B) {
  csi p, j, nz = 0, anz, *Cp, *Ci, *Bp, m, n, bnz, *w, values, *Bi;
  double *x, *Bx, *Cx;
  cs *C;
  printf("cs_multiple_A0\n");
  if (!CS_CSC(A) || !CS_CSC(B))
    return (NULL); /* check inputs */
  printf("cs_multiple_A1\n");
  if (A->n != B->m)
    return (NULL);
  printf("cs_multiple_A\n");
  m = A->m;
  anz = A->p[A->n];
  n = B->n;
  Bp = B->p;
  Bi = B->i;
  Bx = B->x;
  bnz = Bp[n];
  printf("cs_multiple_B\n");
  w = cs_calloc<csi>(m); /* get workspace */
  printf("cs_multiple_C\n");
  values = (A->x != NULL) && (Bx != NULL);
  x = values ? (double *)cs_malloc(sizeof(double) * m)
             : NULL; /* get workspace */
  printf("cs_multiple_D\n");
  C = cs_spalloc(m, n, anz + bnz, values, 0); /* allocate result */
  if (!C || !w || (values && !x))
    return (cs_done(C, w, x, 0));
  Cp = C->p;
  printf("cs_multiple_E\n");
  for (j = 0; j < n; j++) {
    if (nz + m > C->nzmax && !cs_sprealloc(C, 2 * (C->nzmax) + m)) {
      return (cs_done(C, w, x, 0)); /* out of memory */
    }
    Ci = C->i;
    Cx = C->x;  /* C->i and C->x may be reallocated */
    Cp[j] = nz; /* column j of C starts here */
    for (p = Bp[j]; p < Bp[j + 1]; p++) {
      nz = cs_scatter(A, Bi[p], Bx ? Bx[p] : 1, w, x, j + 1, C, nz);
    }
    if (values)
      for (p = Cp[j]; p < nz; p++)
        Cx[p] = x[Ci[p]];
  }
  Cp[n] = nz; /* finalize the last column of C */
  printf("cs_multiple_DONE\n");
  cs_sprealloc(C, 0);           /* remove extra space from C */
  return (cs_done(C, w, x, 1)); /* success; free workspace, return C */
}
