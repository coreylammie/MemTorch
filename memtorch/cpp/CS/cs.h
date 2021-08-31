#ifndef _CS_H
#define _CS_H
#endif
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#define CS_VER 3 /* CSparse Version */
#define CS_SUBVER 1
#define CS_SUBSUB 3
#define CS_DATE "Mar 26, 2014" /* CSparse release date */
#define CS_COPYRIGHT "Copyright (c) Timothy A. Davis, 2006-2014"
#ifndef csi
#define csi ptrdiff_t
#endif
#define CS_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CS_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define CS_FLIP(i) (-(i)-2)
#define CS_UNFLIP(i) (((i) < 0) ? CS_FLIP(i) : (i))
#define CS_MARKED(w, j) (w[j] < 0)
#define CS_MARK(w, j)                                                          \
  { w[j] = CS_FLIP(w[j]); }
#define CS_CSC(A) (A && (A->nz == -1))
#define CS_TRIPLET(A) (A && (A->nz >= 0))
#include "cs_malloc.h"

/* --- primary CSparse routines and data structures ------------------------- */
typedef struct cs_sparse /* matrix in compressed-column or triplet form */
{
  csi nzmax; /* maximum number of entries */
  csi m;     /* number of rows */
  csi n;     /* number of columns */
  csi *p;    /* column pointers (size n+1) or col indices (size nzmax) */
  csi *i;    /* row indices, size nzmax */
  double *x; /* numerical values, size nzmax */
  csi nz;    /* # of entries in triplet matrix, -1 for compressed-col */
} cs;

/* --- secondary CSparse routines and data structures ----------------------- */
typedef struct cs_symbolic /* symbolic Cholesky, LU, or QR analysis */
{
  csi *pinv;     /* inverse row perm. for QR, fill red. perm for Chol */
  csi *q;        /* fill-reducing column permutation for LU and QR */
  csi *parent;   /* elimination tree for Cholesky and QR */
  csi *cp;       /* column pointers for Cholesky, row counts for QR */
  csi *leftmost; /* leftmost[i] = min(find(A(i,:))), for QR */
  csi m2;        /* # of rows for QR, after adding fictitious rows */
  double lnz;    /* # entries in L for LU or Cholesky; in V for QR */
  double unz;    /* # entries in U for LU; in R for QR */
} css;

typedef struct cs_numeric /* numeric Cholesky, LU, or QR factorization */
{
  cs *L;     /* L for LU and Cholesky, V for QR */
  cs *U;     /* U for LU, R for QR, not used for Cholesky */
  csi *pinv; /* partial pivoting for LU */
  double *B; /* beta [0..n-1] for QR */
} csn;

typedef struct cs_dmperm_results /* cs_dmperm or cs_scc output */
{
  csi *p;    /* size m, row permutation */
  csi *q;    /* size n, column permutation */
  csi *r;    /* size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q) */
  csi *s;    /* size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q) */
  csi nb;    /* # of blocks in fine dmperm decomposition */
  csi rr[5]; /* coarse row decomposition */
  csi cc[5]; /* coarse column decomposition */
} csd;

#include "cs_util.h"

#include "cs_cumsum.h"

#include "cs_compress.h"

#include "cs_entry.h"

#include "cs_scatter.h"

#include "cs_transpose.h"

#include "cs_tdfs.h"

#include "cs_permute.h"

#include "cs_etree.h"

#include "cs_post.h"

#include "cs_leaf.h"

#include "cs_counts.h"

#include "cs_fkeep.h"

#include "cs_multiply.h"

#include "cs_add.h"

#include "cs_amd.h"

#include "cs_sqr.h"

#include "cs_ipvec.h"

#include "cs_lsolve.h"

#include "cs_usolve.h"

#include "cs_dfs.h"

#include "cs_reach.h"

#include "cs_spsolve.h"

#include "cs_lu.h"

#include "cs_happly.h"

#include "cs_house.h"

#include "cs_pvec.h"

#include "cs_utsolve.h"

#include "cs_qr.h"

#include "cs_qrsol.h"

#include "cs_lusol.h"