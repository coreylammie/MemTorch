CUDA_CALLABLE_MEMBER
void *cs_realloc(void *p, csi n, size_t size, csi *ok) {
  void *pnew;
  pnew = realloc(p, CS_MAX(n, 1) * size); /* realloc the block */
  *ok = (pnew != NULL);                   /* realloc fails if pnew is NULL */
  return ((*ok) ? pnew : p);              /* return original p if failure */
}

/* allocate a sparse matrix (triplet form or compressed-column form) */
CUDA_CALLABLE_MEMBER
cs *cs_spalloc(csi m, csi n, csi nzmax, csi values, csi triplet) {
  cs *A = (cs *)cs_malloc(sizeof(cs));
  if (!A)
    return NULL;
  A->m = m;
  A->n = n;
  A->nzmax = nzmax = CS_MAX(nzmax, 1);
  A->nz = triplet ? 0 : -1;
  A->p = (ptrdiff_t *)cs_malloc(triplet ? sizeof(csi) * nzmax
                                     : sizeof(csi) * (n + 1));
  A->i = (ptrdiff_t *)cs_malloc(sizeof(csi) * nzmax);
  A->x = values ? (double *)cs_malloc(sizeof(double) * nzmax) : NULL;
  return A;
}

/* change the max # of entries sparse matrix */
CUDA_CALLABLE_MEMBER
csi cs_sprealloc(cs *A, csi nzmax) {
  csi ok, oki = 1, okj = 1, okx = 1;
  if (!A)
    return (0);
  if (nzmax <= 0)
    nzmax = (CS_CSC(A)) ? (A->p[A->n]) : A->nz;

#ifdef __CUDACC__
  free(A->i);
  A->i = (csi *)cs_malloc(sizeof(csi) * nzmax);
  if (CS_TRIPLET(A)) {
    free(A->p);
    A->p = (csi *)cs_malloc(sizeof(csi) * nzmax);
  }
  if (A->x) {
    free(A->x);
    A->x = (double *)cs_malloc(sizeof(double) * nzmax);
  }
#else
  A->i = (ptrdiff_t *)cs_realloc(A->i, nzmax, sizeof(csi), &oki);
  if (CS_TRIPLET(A))
    A->p = (ptrdiff_t *)cs_realloc(A->p, nzmax, sizeof(csi), &okj);
  if (A->x)
    A->x = (double *)cs_realloc(A->x, nzmax, sizeof(double), &okx);
#endif
  ok = (oki && okj && okx);
  // printf("OK: %ld = %ld, %ld, %ld\n", (long)ok, (long)oki, (long)okj,
  //        (long)okx);
  if (ok)
    A->nzmax = nzmax;
  return (ok);
}

/* free a sparse matrix */
CUDA_CALLABLE_MEMBER
cs *cs_spfree(cs *A) {
  if (!A)
    return (NULL); /* do nothing if A already NULL */
  free(A->p);
  free(A->i);
  free(A->x);
  free(A);
  return NULL;
}

/* free a numeric factorization */
CUDA_CALLABLE_MEMBER
csn *cs_nfree(csn *N) {
  if (!N)
    return (NULL); /* do nothing if N already NULL */
  cs_spfree(N->L);
  cs_spfree(N->U);
  free(N->pinv);
  free(N->B);
  free(N);
  return NULL;
}

/* free a symbolic factorization */
CUDA_CALLABLE_MEMBER
css *cs_sfree(css *S) {
  if (!S)
    return (NULL); /* do nothing if S already NULL */
  free(S->pinv);
  free(S->q);
  free(S->parent);
  free(S->cp);
  free(S->leftmost);
  free(S);
  return NULL;
}

/* allocate a cs_dmperm or cs_scc result */
CUDA_CALLABLE_MEMBER
csd *cs_dalloc(csi m, csi n) {
  csd *D;
  D = cs_calloc<csd>(1);
  if (!D)
    return (NULL);
  D->p = (ptrdiff_t *)cs_malloc(sizeof(csi) * m);
  D->r = (ptrdiff_t *)cs_malloc(sizeof(csi) * (m + 6));
  D->q = (ptrdiff_t *)cs_malloc(sizeof(csi) * n);
  D->s = (ptrdiff_t *)cs_malloc(sizeof(csi) * (n + 6));
  return ((!D->p || !D->r || !D->q || !D->s) ? cs_dfree(D) : D);
}

/* free a cs_dmperm or cs_scc result */
CUDA_CALLABLE_MEMBER
csd *cs_dfree(csd *D) {
  if (!D)
    return (NULL); /* do nothing if D already NULL */
  free(D->p);
  free(D->q);
  free(D->r);
  free(D->s);
  free(D);
  return NULL;
}

/* free workspace and return a sparse matrix result */
CUDA_CALLABLE_MEMBER
cs *cs_done(cs *C, void *w, void *x, csi ok) {
  free(w); /* free workspace */
  free(x);
  return (ok ? C : cs_spfree(C)); /* return result if OK, else free it */
}

/* free workspace and return csi array result */
CUDA_CALLABLE_MEMBER
csi *cs_idone(csi *p, cs *C, void *w, csi ok) {
  cs_spfree(C); /* free temporary matrix */
  free(w);      /* free workspace */
  if (!ok) {
    p = NULL;
  }
  return p;
}

/* free workspace and return a numeric factorization (Cholesky, LU, or QR) */
CUDA_CALLABLE_MEMBER
csn *cs_ndone(csn *N, cs *C, void *w, void *x, csi ok) {
  cs_spfree(C); /* free temporary matrix */
  free(w);      /* free workspace */
  free(x);
  return (ok ? N : cs_nfree(N)); /* return result if OK, else free it */
}

/* free workspace and return a csd result */
CUDA_CALLABLE_MEMBER
csd *cs_ddone(csd *D, cs *C, void *w, csi ok) {
  cs_spfree(C);                  /* free temporary matrix */
  free(w);                       /* free workspace */
  return (ok ? D : cs_dfree(D)); /* return result if OK, else free it */
}
