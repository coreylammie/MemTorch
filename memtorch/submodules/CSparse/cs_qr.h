/* sparse QR factorization [V,beta,pinv,R] = qr (A) */
CUDA_CALLABLE_MEMBER
csn *cs_qr(const cs *A, const css *S) {
  double *Rx, *Vx, *Ax, *x, *Beta;
  csi i, k, p, m, n, vnz, p1, top, m2, len, col, rnz, *s, *leftmost, *Ap, *Ai,
      *parent, *Rp, *Ri, *Vp, *Vi, *w, *pinv, *q;
  cs *R, *V;
  csn *N;
  if (!CS_CSC(A) || !S)
    return (NULL);
  m = A->m;
  n = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  q = S->q;
  parent = S->parent;
  pinv = S->pinv;
  m2 = S->m2;
  vnz = S->lnz;
  rnz = S->unz;
  leftmost = S->leftmost;
  w = (csi *)malloc(sizeof(csi) * (m2 + n)); /* get csi workspace */
  x = (double *)malloc(sizeof(double) * m2); /* get double workspace */
  N = cs_calloc<csn>(1);                     /* allocate result */
  printf("cs_qr_INIT\n");
  if (!w || !x || !N)
    return (cs_ndone(N, NULL, w, x, 0));

  printf("cs_qr_A\n");
  s = w + m2; /* s is size n */
  for (k = 0; k < m2; k++)
    x[k] = 0; /* clear workspace x */

  printf("cs_qr_B\n");
  N->L = V = cs_spalloc(m2, n, vnz, 1, 0);            /* allocate result V */
  N->U = R = cs_spalloc(m2, n, rnz, 1, 0);            /* allocate result R */
  N->B = Beta = (double *)malloc(sizeof(double) * n); /* allocate result Beta */
  if (!R || !V || !Beta)
    return (cs_ndone(N, NULL, w, x, 0));

  printf("cs_qr_C\n");
  Rp = R->p;
  Ri = R->i;
  Rx = R->x;
  Vp = V->p;
  Vi = V->i;
  // printf("%ld.\n", (long)Vi->nnz);
  Vx = V->x;
  for (i = 0; i < m2; i++)
    w[i] = -1; /* clear w, to mark nodes */
  rnz = 0;
  vnz = 0;
  printf("cs_qr_D\n");
  for (k = 0; k < n; k++) /* compute V and R */
  {
    Rp[k] = rnz; /* R(:,k) starts here */
    printf("cs_qr_D_1\n");
    Vp[k] = p1 = vnz; /* V(:,k) starts here */
    printf("cs_qr_D_2\n");
    w[k] = k; /* add V(k,k) to pattern of V */
    printf("cs_qr_D_3\n");
    Vi[vnz++] = k;
    printf("cs_qr_D_4\n");
    top = n;
    printf("cs_qr_D_5\n");
    col = q ? q[k] : k;
    printf("cs_qr_D_6\n");
    for (p = Ap[col]; p < Ap[col + 1]; p++) /* find R(:,k) pattern */
    {
      printf("cs_leftmost_1\n");
      i = leftmost[Ai[p]]; /* i = min(find(A(i,q))) */
      printf("cs_leftmost_2\n");
      for (len = 0; w[i] != k; i = parent[i]) /* traverse up to k */
      {
        s[len++] = i;
        w[i] = k;
      }
      while (len > 0)
        s[--top] = s[--len]; /* push path on stack */
      i = pinv[Ai[p]];       /* i = permuted row of A(:,col) */
      x[i] = Ax[p];          /* x (i) = A(:,col) */
      if (i > k && w[i] < k) /* pattern of V(:,k) = x (k+1:m) */
      {
        Vi[vnz++] = i; /* add i to pattern of V(:,k) */
        w[i] = k;
      }
    }
    for (p = top; p < n; p++) /* for each i in pattern of R(:,k) */
    {
      i = s[p]; /* R(i,k) is nonzero */
      printf("cs_qr_happily_1\n");
      cs_happly(V, i, Beta[i], x); /* apply (V(i),Beta(i)) to x */
      printf("cs_qr_happily_2\n");
      Ri[rnz] = i; /* R(i,k) = x(i) */
      Rx[rnz++] = x[i];
      x[i] = 0;
      if (parent[i] == k) {
        printf("cs_scatter_1\n");
        vnz = cs_scatter(V, i, 0, w, NULL, k, V, vnz);
        printf("cs_scatter_2\n");
      }
    }
    for (p = p1; p < vnz; p++) /* gather V(:,k) = x */
    {
      Vx[p] = x[Vi[p]];
      x[Vi[p]] = 0;
    }
    Ri[rnz] = k; /* R(k,k) = norm (x) */
    printf("cs_qr_E1\n");
    Rx[rnz++] = cs_house(Vx + p1, Beta + k, vnz - p1); /* [v,beta]=house(x) */
    printf("cs_qr_E2\n");
  }
  printf("cs_qr_N-1\n");
  Rp[n] = rnz; /* finalize R */
  Vp[n] = vnz; /* finalize V */
  printf("cs_qr_DONE\n");
  return (cs_ndone(N, NULL, w, x, 1)); /* success */
}
