#pragma once
#ifndef CS_H
#define CS_H
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER inline __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER inline
#endif

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

CUDA_CALLABLE_MEMBER
void *cs_realloc(void *p, csi n, size_t size, csi *ok);

template <class T> CUDA_CALLABLE_MEMBER T *cs_calloc(size_t n) {
  T *p = (T *)malloc(sizeof(T) * n);
  memset(p, 0, sizeof(T) * n);
  return p;
}

template <class T> CUDA_CALLABLE_MEMBER void swap(T &a, T &b) {
  T c(a);
  a = b;
  b = c;
}

/* --- primary CSparse routines and data structures
   ------------------------- */
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

CUDA_CALLABLE_MEMBER
cs *cs_spalloc(csi m, csi n, csi nzmax, csi values, csi triplet);
CUDA_CALLABLE_MEMBER
csi cs_sprealloc(cs *A, csi nzmax);
CUDA_CALLABLE_MEMBER
cs *cs_spfree(cs *A);
CUDA_CALLABLE_MEMBER
csn *cs_nfree(csn *N);
CUDA_CALLABLE_MEMBER
css *cs_sfree(css *S);
CUDA_CALLABLE_MEMBER
csd *cs_dalloc(csi m, csi n);
CUDA_CALLABLE_MEMBER
csd *cs_dfree(csd *D);
CUDA_CALLABLE_MEMBER
cs *cs_done(cs *C, void *w, void *x, csi ok);
CUDA_CALLABLE_MEMBER
csi *cs_idone(csi *p, cs *C, void *w, csi ok);
CUDA_CALLABLE_MEMBER
csn *cs_ndone(csn *N, cs *C, void *w, void *x, csi ok);
CUDA_CALLABLE_MEMBER
csd *cs_ddone(csd *D, cs *C, void *w, csi ok);
CUDA_CALLABLE_MEMBER
double cs_cumsum(csi *p, csi *c, csi n);
CUDA_CALLABLE_MEMBER
cs *cs_compress(const cs *T);
CUDA_CALLABLE_MEMBER
csi cs_entry(cs *T, csi i, csi j, double x);
CUDA_CALLABLE_MEMBER
csi cs_scatter(const cs *A, csi j, double beta, csi *w, double *x, csi mark,
               cs *C, csi nz);
CUDA_CALLABLE_MEMBER
cs *cs_transpose(const cs *A, csi values);
CUDA_CALLABLE_MEMBER
csi cs_tdfs(csi j, csi k, csi *head, const csi *next, csi *post, csi *stack);
CUDA_CALLABLE_MEMBER
cs *cs_permute(const cs *A, const csi *pinv, const csi *q, csi values);
CUDA_CALLABLE_MEMBER
csi *cs_etree(const cs *A, csi ata);
CUDA_CALLABLE_MEMBER
csi *cs_post(const csi *parent, csi n);
CUDA_CALLABLE_MEMBER
csi cs_leaf(csi i, csi j, const csi *first, csi *maxfirst, csi *prevleaf,
            csi *ancestor, csi *jleaf);
CUDA_CALLABLE_MEMBER
void init_ata(cs *AT, const csi *post, csi *w, csi **head, csi **next);
CUDA_CALLABLE_MEMBER
csi *cs_counts(const cs *A, const csi *parent, const csi *post, csi ata);
CUDA_CALLABLE_MEMBER
csi cs_fkeep(cs *A, csi (*fkeep)(csi, csi, double, void *), void *other);
CUDA_CALLABLE_MEMBER
cs *cs_multiply(const cs *A, const cs *B);
CUDA_CALLABLE_MEMBER
cs *cs_add(const cs *A, const cs *B, double alpha, double beta);
CUDA_CALLABLE_MEMBER
csi cs_wclear(csi mark, csi lemax, csi *w, csi n);
CUDA_CALLABLE_MEMBER
csi cs_diag(csi i, csi j, double aij, void *other);
CUDA_CALLABLE_MEMBER
csi *cs_amd(csi order, const cs *A);
CUDA_CALLABLE_MEMBER
csi cs_vcount(const cs *A, css *S);
CUDA_CALLABLE_MEMBER
css *cs_sqr(csi order, const cs *A, csi qr);
CUDA_CALLABLE_MEMBER
csi cs_ipvec(const csi *p, const double *b, double *x, csi n);
CUDA_CALLABLE_MEMBER
csi cs_usolve(const cs *U, double *x);
CUDA_CALLABLE_MEMBER
csi cs_dfs(csi j, cs *G, csi top, csi *xi, csi *pstack, const csi *pinv);
CUDA_CALLABLE_MEMBER
csi cs_reach(cs *G, const cs *B, csi k, csi *xi, const csi *pinv);
CUDA_CALLABLE_MEMBER
csi cs_happly(const cs *V, csi i, double beta, double *x);
CUDA_CALLABLE_MEMBER
double cs_house(double *x, double *beta, csi n);
CUDA_CALLABLE_MEMBER
csi cs_pvec(const csi *p, const double *b, double *x, csi n);
CUDA_CALLABLE_MEMBER
csi cs_utsolve(const cs *U, double *x);
CUDA_CALLABLE_MEMBER
csn *cs_qr(const cs *A, const css *S);
CUDA_CALLABLE_MEMBER
csi cs_qrsol(csi order, const cs *A, double *b);

#include "cs_add.h"
#include "cs_amd.h"
#include "cs_compress.h"
#include "cs_counts.h"
#include "cs_cumsum.h"
#include "cs_dfs.h"
#include "cs_entry.h"
#include "cs_etree.h"
#include "cs_fkeep.h"
#include "cs_happly.h"
#include "cs_house.h"
#include "cs_ipvec.h"
#include "cs_leaf.h"
#include "cs_multiply.h"
#include "cs_permute.h"
#include "cs_post.h"
#include "cs_pvec.h"
#include "cs_qr.h"
#include "cs_qrsol.h"
#include "cs_reach.h"
#include "cs_scatter.h"
#include "cs_sqr.h"
#include "cs_tdfs.h"
#include "cs_transpose.h"
#include "cs_usolve.h"
#include "cs_util.h"
#include "cs_utsolve.h"
#endif