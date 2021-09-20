/* clear w */
CUDA_CALLABLE_MEMBER
csi cs_wclear(csi mark, csi lemax, csi *w, csi n) {
  csi k;
  if (mark < 2 || (mark + lemax < 0)) {
    for (k = 0; k < n; k++)
      if (w[k] != 0)
        w[k] = 1;
    mark = 2;
  }
  return (mark); /* at this point, w [0..n-1] < mark holds */
}

/* keep off-diagonal entries; drop diagonal entries */
CUDA_CALLABLE_MEMBER
csi cs_diag(csi i, csi j, double aij, void *other) { return (i != j); }

/* p = amd(A+A') if symmetric is true, or amd(A'A) otherwise */
CUDA_CALLABLE_MEMBER
csi *cs_amd(csi order, const cs *A) /* order 0:natural, 1:Chol, 2:LU, 3:QR */
{
  cs *C, *A2, *AT;
  csi *Cp, *Ci, *last, *W, *len, *nv, *next, *P, *head, *elen, *degree, *w,
      *hhead, *ATp, *ATi, d, dk, dext,
      lemax = 0, e, elenk, eln, i, j, k, k1, k2, k3, jlast, ln, dense, nzmax,
      mindeg = 0, nvi, nvj, nvk, mark, wnvi, ok, cnz, nel = 0, p, p1, p2, p3,
      p4, pj, pk, pk1, pk2, pn, q, n, m, t;
  csi h;
  /* --- Construct matrix C ----------------------------------------------- */
  printf("cs_amd_start.\n");
  if (!CS_CSC(A) || order <= 0 || order > 3)
    return (NULL); /* check */

  printf("cs_transpose_a.\n");
  AT = cs_transpose(A, 0); /* compute A' */
  printf("cs_transpose_B.\n");
  if (!AT)
    return (NULL);

  printf("AT->nz = %ld\n", (long)AT->nz);
  printf("cs_transpose_C.\n");
  m = A->m;
  n = A->n;
  dense = CS_MAX(16, 10 * sqrt((double)n)); /* find dense threshold */
  dense = CS_MIN(n - 2, dense);
  if (order == 1 && n == m) {
    printf("cs_order_A.\n");
    C = cs_add(A, AT, 0, 0); /* C = A+A' */
    printf("cs_order_B.\n");
  } else if (order == 2) {
    ATp = &(AT->p[0]); /* drop dense columns from AT */
    ATi = &(AT->i[0]);
    for (p2 = 0, j = 0; j < m; j++) {
      p = ATp[j];  /* column j of AT starts here */
      ATp[j] = p2; /* new column j starts here */
      if (ATp[j + 1] - p > dense)
        continue; /* skip dense col j */
      for (; p < ATp[j + 1]; p++)
        ATi[p2++] = ATi[p];
    }
    ATp[m] = p2;                         /* finalize AT */
    A2 = cs_transpose(AT, 0);            /* A2 = AT' */
    C = A2 ? cs_multiply(AT, A2) : NULL; /* C=A'*A with no dense rows */
    cs_spfree(A2);
  } else {
    C = cs_multiply(AT, A); /* C=A'*A */
  }
  printf("cs_order_C.\n");
  // cs_spfree(AT);
  printf("cs_order_C1.\n");
  if (!C) {
    printf("C is NULL.\n");
    return (NULL);
  }
  printf("cs_order_C2.\n");
  cs_fkeep(C, &cs_diag, NULL); /* drop diagonal entries */
  printf("cs_order_D.\n");
  Cp = &(C->p[0]);
  printf("cs_order_E.\n");
  cnz = Cp[n];
  P = (ptrdiff_t *)cs_malloc(sizeof(csi) * (n + 1));       /* allocate result */
  W = (ptrdiff_t *)cs_malloc(sizeof(csi) * (8 * (n + 1))); /* get workspace */
  printf("cs_order_F.\n");
  t = cnz + cnz / 5 + 2 * n; /* add elbow room to C */
  if (!P || !W || !cs_sprealloc(C, t)) {
    return (cs_idone(P, C, W, 0));
  }
  len = W;
  nv = W + (n + 1);
  next = W + 2 * (n + 1);
  head = W + 3 * (n + 1);
  elen = W + 4 * (n + 1);
  degree = W + 5 * (n + 1);
  w = W + 6 * (n + 1);
  hhead = W + 7 * (n + 1);
  last = P; /* use P as workspace for last */
  printf("cs_amd_quotient.\n");
  /* --- Initialize quotient graph ---------------------------------------- */
  for (k = 0; k < n; k++)
    len[k] = Cp[k + 1] - Cp[k];
  len[n] = 0;
  nzmax = C->nzmax;
  Ci = &(C->i[0]);
  for (i = 0; i <= n; i++) {
    head[i] = -1; /* degree list i is empty */
    last[i] = -1;
    next[i] = -1;
    hhead[i] = -1;      /* hash list i is empty */
    nv[i] = 1;          /* node i is just one node */
    w[i] = 1;           /* node i is alive */
    elen[i] = 0;        /* Ek of node i is empty */
    degree[i] = len[i]; /* degree of node i */
  }
  mark = cs_wclear(0, 0, w, n); /* clear w */
  elen[n] = -2;                 /* n is a dead element */
  Cp[n] = -1;                   /* n is a root of assembly tree */
  w[n] = 0;                     /* n is a dead element */
  printf("cs_amd_degree.\n");
  /* --- Initialize degree lists ------------------------------------------ */
  for (i = 0; i < n; i++) {
    d = degree[i];
    if (d == 0) /* node i is empty */
    {
      elen[i] = -2; /* element i is dead */
      nel++;
      Cp[i] = -1; /* i is a root of assembly tree */
      w[i] = 0;
    } else if (d > dense) /* node i is dense */
    {
      nv[i] = 0;    /* absorb i into element n */
      elen[i] = -1; /* node i is dead */
      nel++;
      Cp[i] = CS_FLIP(n);
      nv[n]++;
    } else {
      if (head[d] != -1)
        last[head[d]] = i;
      next[i] = head[d]; /* put node i in degree list d */
      head[d] = i;
    }
  }
  printf("cs_amd_select_pivots.\n");
  while (nel < n) /* while (selecting pivots) do */
  {
    /* --- Select node of minimum approximate degree -------------------- */
    for (k = -1; mindeg < n && (k = head[mindeg]) == -1; mindeg++)
      ;
    if (next[k] != -1)
      last[next[k]] = -1;
    head[mindeg] = next[k]; /* remove k from degree list */
    elenk = elen[k];        /* elenk = |Ek| */
    nvk = nv[k];            /* # of nodes k represents */
    nel += nvk;             /* nv[k] nodes of A eliminated */
    /* --- Garbage collection ------------------------------------------- */
    if (elenk > 0 && cnz + mindeg >= nzmax) {
      for (j = 0; j < n; j++) {
        if ((p = Cp[j]) >= 0) /* j is a live node or element */
        {
          Cp[j] = Ci[p];      /* save first entry of object */
          Ci[p] = CS_FLIP(j); /* first entry is now CS_FLIP(j) */
        }
      }
      for (q = 0, p = 0; p < cnz;) /* scan all of memory */
      {
        if ((j = CS_FLIP(Ci[p++])) >= 0) /* found object j */
        {
          Ci[q] = Cp[j]; /* restore first entry of object */
          Cp[j] = q++;   /* new pointer to object j */
          for (k3 = 0; k3 < len[j] - 1; k3++)
            Ci[q++] = Ci[p++];
        }
      }
      cnz = q; /* Ci [cnz...nzmax-1] now free */
    }
    printf("cs_amd_construct_new_element.\n");
    /* --- Construct new element ---------------------------------------- */
    dk = 0;
    nv[k] = -nvk; /* flag k as in Lk */
    p = Cp[k];
    pk1 = (elenk == 0) ? p : cnz; /* do in place if elen[k] == 0 */
    pk2 = pk1;
    printf("cs_amd_construct_new_element_A.\n");
    for (k1 = 1; k1 <= elenk + 1; k1++) {
      printf("cs_amd_construct_new_element_B.\n");
      if (k1 > elenk) {
        e = k;               /* search the nodes in k */
        pj = p;              /* list of nodes starts at Ci[pj]*/
        ln = len[k] - elenk; /* length of list of nodes in k */
      } else {
        e = Ci[p++]; /* search the nodes in e */
        pj = Cp[e];
        ln = len[e]; /* length of list of nodes in e */
      }
      printf("cs_amd_construct_new_element_C.\n");
      for (k2 = 1; k2 <= ln; k2++) {
        printf("cs_amd_construct_new_element_C0.\n");
        i = Ci[pj++];
        printf("cs_amd_construct_new_element_C1.\n");

        printf("i = %lld\n", (long long)i);
        printf("nvi = %lld\n", (long long)nvi);
        // printf("nv[i] = %ld\n", (long long)nv[i]);
        // if ((i > (n + 1)) || ((nvi = nv[i]) <= 0)) {
        //   continue; /* node i dead, or seen */
        // }
        // if ((i >= (n + 1)) || (i < 0)) {
        //   i = 0;
        // }
        if ((nvi = nv[i]) <= 0) {
          continue; /* node i dead, or seen */
        }
        printf("cs_amd_construct_new_element_C2.\n");
        dk += nvi; /* degree[Lk] += size of node i */
        printf("cs_amd_construct_new_element_C3.\n");
        nv[i] = -nvi; /* negate nv[i] to denote i in Lk*/
        printf("cs_amd_construct_new_element_C4.\n");
        Ci[pk2++] = i; /* place i in Lk */
        printf("cs_amd_construct_new_element_C5.\n");
        if (next[i] != -1)
          last[next[i]] = last[i];

        printf("cs_amd_construct_new_element_C6.\n");
        if (last[i] != -1) /* remove i from degree list */
        {
          printf("cs_amd_construct_new_element_C7A.\n");
          next[last[i]] = next[i];
        } else {
          printf("cs_amd_construct_new_element_CB.\n");
          head[degree[i]] = next[i];
        }
      }
      printf("cs_amd_construct_new_element_D.\n");
      if (e != k) {
        Cp[e] = CS_FLIP(k); /* absorb e into k */
        w[e] = 0;           /* e is now a dead element */
      }
    }
    printf("cs_amd_construct_new_element_E.\n");
    if (elenk != 0)
      cnz = pk2;    /* Ci [cnz...nzmax] is free */
    degree[k] = dk; /* external degree of k - |Lk\i| */
    Cp[k] = pk1;    /* element k is in Ci[pk1..pk2-1] */
    len[k] = pk2 - pk1;
    elen[k] = -2; /* k is now an element */
    printf("cs_amd_find_set_differences.\n");
    /* --- Find set differences ----------------------------------------- */
    mark = cs_wclear(mark, lemax, w, n); /* clear w if necessary */
    for (pk = pk1; pk < pk2; pk++)       /* scan 1: find |Le\Lk| */
    {
      i = Ci[pk];
      // if ((i >= (n + 1)) || (i < 0)) {
      //   i = 0;
      // }
      if ((eln = elen[i]) <= 0)
        continue;   /* skip if elen[i] empty */
      nvi = -nv[i]; /* nv [i] was negated */
      wnvi = mark - nvi;
      for (p = Cp[i]; p <= Cp[i] + eln - 1; p++) /* scan Ei */
      {
        e = Ci[p];
        // if ((e >= (n + 1)) || (e < 0)) {
        //   e = 0;
        // }
        if (w[e] >= mark) {
          w[e] -= nvi;        /* decrement |Le\Lk| */
        } else if (w[e] != 0) /* ensure e is a live element */
        {
          w[e] = degree[e] + wnvi; /* 1st time e seen in scan 1 */
        }
      }
    }
    printf("cs_amd_degree_update.\n");
    /* --- Degree update ------------------------------------------------ */
    for (pk = pk1; pk < pk2; pk++) /* scan2: degree update */
    {
      printf("cs_amd_degree_update_A.\n");
      printf("pk = %d\n", (long long)pk);
      i = Ci[pk]; /* consider node i in Lk */
      p1 = Cp[i];
      printf("cs_amd_degree_update_B1.\n");
      p2 = p1 + elen[i] - 1;
      printf("cs_amd_degree_update_B2.\n");
      pn = p1;
      printf("cs_amd_degree_update_B3.\n");
      for (h = 0, d = 0, p = p1; p <= p2; p++) /* scan Ei */
      {
        e = Ci[p];
        printf("cs_amd_degree_update_B4.\n");
        printf("p = %lld\n", p);
        printf("e = %lld\n", e);

        if (w[e] != 0) /* e is an unabsorbed element */
        {
          printf("cs_amd_degree_update_B5.\n");
          dext = w[e] - mark; /* dext = |Le\Lk| */
          printf("cs_amd_degree_update_B6.\n");
          if (dext > 0) {
            printf("cs_amd_degree_update_B7.\n");
            d += dext; /* sum up the set differences */
            printf("cs_amd_degree_update_B8.\n");
            Ci[pn++] = e; /* keep e in Ei */
            printf("cs_amd_degree_update_B9.\n");
            h += e; /* compute the hash of node i */
          } else {
            Cp[e] = CS_FLIP(k); /* aggressive absorb. e->k */
            w[e] = 0;           /* e is a dead element */
          }
        }
      }
      printf("cs_amd_degree_update_C1.\n");
      elen[i] = pn - p1 + 1; /* elen[i] = |Ei| */
      printf("cs_amd_degree_update_C2.\n");
      p3 = pn;
      printf("cs_amd_degree_update_C3.\n");
      p4 = p1 + len[i];
      printf("cs_amd_degree_update_C4.\n");
      for (p = p2 + 1; p < p4; p++) /* prune edges in Ai */
      {
        j = Ci[p];
        printf("cs_amd_degree_update_C5A.\n");
        printf("j = %lld\n", (long long)j);
        // if ((j >= (n + 1)) || (j < 0)) {
        //   j = 0;
        // }
        printf("cs_amd_degree_update_C5B.\n");
        if ((nvj = nv[j]) <= 0)
          continue; /* node j dead or in Lk */

        printf("cs_amd_degree_update_C6.\n");
        d += nvj; /* degree(i) += |j| */
        printf("cs_amd_degree_update_C7.\n");
        Ci[pn++] = j; /* place j in node list of i */
        printf("cs_amd_degree_update_C8.\n");
        h += j; /* compute hash for node i */
      }
      printf("cs_amd_degree_update_D.\n");
      if (d == 0) /* check for mass elimination */
      {
        Cp[i] = CS_FLIP(k); /* absorb i into k */
        nvi = -nv[i];
        dk -= nvi;  /* |Lk| -= |i| */
        nvk += nvi; /* |k| += nv[i] */
        nel += nvi;
        nv[i] = 0;
        elen[i] = -1; /* node i is dead */
      } else {
        degree[i] = CS_MIN(degree[i], d); /* update degree(i) */
        Ci[pn] = Ci[p3];                  /* move first node to end */
        Ci[p3] = Ci[p1];                  /* move 1st el. to end of Ei */
        Ci[p1] = k;                       /* add k as 1st element in of Ei */
        len[i] = pn - p1 + 1;             /* new len of adj. list of node i */
        h = ((h < 0) ? (-h) : h) % n;     /* finalize hash of i */
        next[i] = hhead[h];               /* place i in hash bucket */
        hhead[h] = i;
        last[i] = h; /* save hash of i in last[i] */
      }
    } /* scan2 is done */
    printf("cs_amd_degree_update_E.\n");
    degree[k] = dk; /* finalize |Lk| */
    lemax = CS_MAX(lemax, dk);
    mark = cs_wclear(mark + lemax, lemax, w, n); /* clear w */
    printf("cs_amd_supernode_detection.\n");
    /* --- Supernode detection ------------------------------------------ */
    for (pk = pk1; pk < pk2; pk++) {
      i = Ci[pk];
      if (nv[i] >= 0)
        continue;  /* skip if i is dead */
      h = last[i]; /* scan hash bucket of node i */
      i = hhead[h];
      hhead[h] = -1; /* hash bucket will be empty */
      for (; i != -1 && next[i] != -1; i = next[i], mark++) {
        ln = len[i];
        eln = elen[i];
        for (p = Cp[i] + 1; p <= Cp[i] + ln - 1; p++)
          w[Ci[p]] = mark;
        jlast = i;
        for (j = next[i]; j != -1;) /* compare i with all j */
        {
          ok = (len[j] == ln) && (elen[j] == eln);
          for (p = Cp[j] + 1; ok && p <= Cp[j] + ln - 1; p++) {
            if (w[Ci[p]] != mark)
              ok = 0; /* compare i and j*/
          }
          if (ok) /* i and j are identical */
          {
            Cp[j] = CS_FLIP(i); /* absorb j into i */
            nv[i] += nv[j];
            nv[j] = 0;
            elen[j] = -1; /* node j is dead */
            j = next[j];  /* delete j from hash bucket */
            next[jlast] = j;
          } else {
            jlast = j; /* j and i are different */
            j = next[j];
          }
        }
      }
    }
    printf("cs_amd_finalize_new_element.\n");
    /* --- Finalize new element------------------------------------------ */
    for (p = pk1, pk = pk1; pk < pk2; pk++) /* finalize Lk */
    {
      i = Ci[pk];
      if ((nvi = -nv[i]) <= 0)
        continue;               /* skip if i is dead */
      nv[i] = nvi;              /* restore nv[i] */
      d = degree[i] + dk - nvi; /* compute external degree(i) */
      d = CS_MIN(d, n - nel - nvi);
      if (head[d] != -1)
        last[head[d]] = i;
      next[i] = head[d]; /* put i back in degree list */
      last[i] = -1;
      head[d] = i;
      mindeg = CS_MIN(mindeg, d); /* find new minimum degree */
      degree[i] = d;
      Ci[p++] = i; /* place i in Lk */
    }
    nv[k] = nvk;                 /* # nodes absorbed into k */
    if ((len[k] = p - pk1) == 0) /* length of adj list of element k*/
    {
      Cp[k] = -1; /* k is a root of the tree */
      w[k] = 0;   /* k is now a dead element */
    }
    if (elenk != 0)
      cnz = p; /* free unused space in Lk */
  }
  printf("cs_amd_postordering.\n");
  /* --- Postordering ----------------------------------------------------- */
  for (i = 0; i < n; i++)
    Cp[i] = CS_FLIP(Cp[i]); /* fix assembly tree */
  for (j = 0; j <= n; j++)
    head[j] = -1;
  for (j = n; j >= 0; j--) /* place unordered nodes in lists */
  {
    if (nv[j] > 0)
      continue;            /* skip if j is an element */
    next[j] = head[Cp[j]]; /* place j in list of its parent */
    head[Cp[j]] = j;
  }
  for (e = n; e >= 0; e--) /* place elements in lists */
  {
    if (nv[e] <= 0)
      continue; /* skip unless e is an element */
    if (Cp[e] != -1) {
      next[e] = head[Cp[e]]; /* place e in list of its parent */
      head[Cp[e]] = e;
    }
  }
  for (k = 0, i = 0; i <= n; i++) /* postorder the assembly tree */
  {
    if (Cp[i] == -1)
      k = cs_tdfs(i, k, head, next, P, w);
  }
  printf("cs_amd_end.\n");
  return (cs_idone(P, C, W, 1));
}
