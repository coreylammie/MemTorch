#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

// #include "superlu/slu_ddefs.h"

#include "solve_passive.h"

// void get_perm_c(int ispec, SuperMatrix *A, int *perm_c) {
//   NCformat *Astore = A->Store;
//   int m, n, bnz = 0, *b_colptr, i;
//   int delta, maxint, nofsub, *invp;
//   int *b_rowind, *dhead, *qsize, *llist, *marker;
//   double t, SuperLU_timer_();

//   m = A->nrow;
//   n = A->ncol;

//   t = SuperLU_timer_();
//   switch (ispec) {
//   case (NATURAL): /* Natural ordering */
//     for (i = 0; i < n; ++i)
//       perm_c[i] = i;
// #if (PRNTlevel >= 1)
//     printf("Use natural column ordering.\n");
// #endif
//     return;
//   case (MMD_ATA): /* Minimum degree ordering on A'*A */
//     getata(m, n, Astore->nnz, Astore->colptr, Astore->rowind, &bnz,
//     &b_colptr,
//            &b_rowind);
// #if (PRNTlevel >= 1)
//     printf("Use minimum degree ordering on A'*A.\n");
// #endif
//     t = SuperLU_timer_() - t;
//     /*printf("Form A'*A time = %8.3f\n", t);*/
//     break;
//   case (MMD_AT_PLUS_A): /* Minimum degree ordering on A'+A */
//     if (m != n)
//       ABORT("Matrix is not square");
//     at_plus_a(n, Astore->nnz, Astore->colptr, Astore->rowind, &bnz,
//     &b_colptr,
//               &b_rowind);
// #if (PRNTlevel >= 1)
//     printf("Use minimum degree ordering on A'+A.\n");
// #endif
//     t = SuperLU_timer_() - t;
//     /*printf("Form A'+A time = %8.3f\n", t);*/
//     break;
//   case (COLAMD): /* Approximate minimum degree column ordering. */
//     get_colamd(m, n, Astore->nnz, Astore->colptr, Astore->rowind, perm_c);
// #if (PRNTlevel >= 1)
//     printf(".. Use approximate minimum degree column ordering.\n");
// #endif
//     return;
//   default:
//     ABORT("Invalid ISPEC");
//   }

//   if (bnz != 0) {
//     t = SuperLU_timer_();

//     /* Initialize and allocate storage for GENMMD. */
//     delta = 0;           /* DELTA is a parameter to allow the choice of nodes
//                             whose degree <= min-degree + DELTA. */
//     maxint = 2147483647; /* 2**31 - 1 */
//     invp = (int *)SUPERLU_MALLOC((n + delta) * sizeof(int));
//     if (!invp)
//       ABORT("SUPERLU_MALLOC fails for invp.");
//     dhead = (int *)SUPERLU_MALLOC((n + delta) * sizeof(int));
//     if (!dhead)
//       ABORT("SUPERLU_MALLOC fails for dhead.");
//     qsize = (int *)SUPERLU_MALLOC((n + delta) * sizeof(int));
//     if (!qsize)
//       ABORT("SUPERLU_MALLOC fails for qsize.");
//     llist = (int *)SUPERLU_MALLOC(n * sizeof(int));
//     if (!llist)
//       ABORT("SUPERLU_MALLOC fails for llist.");
//     marker = (int *)SUPERLU_MALLOC(n * sizeof(int));
//     if (!marker)
//       ABORT("SUPERLU_MALLOC fails for marker.");

//     /* Transform adjacency list into 1-based indexing required by GENMMD.*/
//     for (i = 0; i <= n; ++i)
//       ++b_colptr[i];
//     for (i = 0; i < bnz; ++i)
//       ++b_rowind[i];

//     genmmd_(&n, b_colptr, b_rowind, perm_c, invp, &delta, dhead, qsize,
//     llist,
//             marker, &maxint, &nofsub);

//     /* Transform perm_c into 0-based indexing. */
//     for (i = 0; i < n; ++i)
//       --perm_c[i];

//     SUPERLU_FREE(invp);
//     SUPERLU_FREE(dhead);
//     SUPERLU_FREE(qsize);
//     SUPERLU_FREE(llist);
//     SUPERLU_FREE(marker);
//     SUPERLU_FREE(b_rowind);

//     t = SuperLU_timer_() - t;
//     /*  printf("call GENMMD time = %8.3f\n", t);*/

//   } else { /* Empty adjacency structure */
//     for (i = 0; i < n; ++i)
//       perm_c[i] = i;
//   }

//   SUPERLU_FREE(b_colptr);
// }

// __host__ __device__ void dgssv(superlu_options_t *options, SuperMatrix *A,
//                                int *perm_c, int *perm_r, SuperMatrix *L,
//                                SuperMatrix *U, SuperMatrix *B,
//                                SuperLUStat_t *stat, int *info) {

//   DNformat *Bstore;
//   SuperMatrix *AA; /* A in SLU_NC format used by the factorization routine.*/
//   SuperMatrix AC;  /* Matrix postmultiplied by Pc */
//   int lwork = 0, *etree, i;
//   GlobalLU_t Glu; /* Not needed on return. */
//   /* Set default values for some parameters */
//   int panel_size; /* panel size */
//   int relax;      /* no of columns in a relaxed snodes */
//   int permc_spec;
//   trans_t trans = NOTRANS;
//   /* Test the input parameters ... */
//   *info = 0;
//   Bstore = (DNformat *)B->Store;
//   if (options->Fact != DOFACT)
//     *info = -1;
//   else if (A->nrow != A->ncol || A->nrow < 0 ||
//            (A->Stype != SLU_NC && A->Stype != SLU_NR) || A->Dtype != SLU_D ||
//            A->Mtype != SLU_GE)
//     *info = -2;
//   else if (B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
//            B->Stype != SLU_DN || B->Dtype != SLU_D || B->Mtype != SLU_GE)
//     *info = -7;
//   if (*info != 0) {
//     i = -(*info);
//     return;
//   }
//   /* Convert A to SLU_NC format when necessary. */
//   if (A->Stype == SLU_NR) {
//     NRformat *Astore = (NRformat *)A->Store;
//     AA = (SuperMatrix *)malloc(sizeof(SuperMatrix));
//     dCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
//                            (double *)Astore->nzval, Astore->colind,
//                            Astore->rowptr, SLU_NC, A->Dtype, A->Mtype);
//     trans = TRANS;
//   } else {
//     if (A->Stype == SLU_NC)
//       AA = A;
//   }
//   /*
//    * Get column permutation vector perm_c[], according to permc_spec:
//    *   permc_spec = NATURAL:  natural ordering
//    *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
//    *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
//    *   permc_spec = COLAMD:   approximate minimum degree column ordering
//    *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
//    */
//   permc_spec = options->ColPerm;
//   if (permc_spec != MY_PERMC && options->Fact == DOFACT)
//     get_perm_c(permc_spec, AA, perm_c);

//   etree = (int *)malloc(sizeof(int) * A->ncol);
//   sp_preorder(options, AA, perm_c, etree, &AC);
//   panel_size = sp_ienv(1);
//   relax = sp_ienv(2);
//   /* Compute the LU factorization of A. */
//   dgstrf(options, &AC, relax, panel_size, etree, NULL, lwork, perm_c, perm_r,
//   L,
//          U, &Glu, stat, info);
//   if (*info == 0) {
//     /* Solve the system A*X=B, overwriting B with X. */
//     dgstrs(trans, L, U, perm_c, perm_r, B, stat, info);
//   }
//   // SUPERLU_FREE(etree);
//   // Destroy_CompCol_Permuted(&AC);
//   // if (A->Stype == SLU_NR) {
//   //   Destroy_SuperMatrix_Store(AA);
//   //   // SUPERLU_FREE(AA);
//   // }
// }

// __host__ __device__ int sp_ienv(int ispec) {
//   int i;
//   switch (ispec) {
//   case 1:
//     return (20);
//   case 2:
//     return (10);
//   case 3:
//     return (200);
//   case 4:
//     return (200);
//   case 5:
//     return (100);
//   case 6:
//     return (30);
//   case 7:
//     return (10);
//   }
//   /* Invalid value for ISPEC */
//   i = 1;
//   return 0;

// } /* sp_ienv_ */

// __host__ __device__ int *intCalloc(int n) {
//   int *buf;
//   register int i;
//   buf = (int *)malloc(n * sizeof(int));
//   for (i = 0; i < n; ++i)
//     buf[i] = 0;
//   return (buf);
// }

// __host__ __device__ void StatInit(SuperLUStat_t *stat) {
//   register int i, w, panel_size, relax;
//   panel_size = sp_ienv(1);
//   relax = sp_ienv(2);
//   w = SUPERLU_MAX(panel_size, relax);
//   stat->panel_histo = intCalloc(w + 1);
//   stat->utime = (double *)malloc(NPHASES * sizeof(double));
//   stat->ops = (flops_t *)malloc(NPHASES * sizeof(flops_t));
//   for (i = 0; i < NPHASES; ++i) {
//     stat->utime[i] = 0.;
//     stat->ops[i] = 0.;
//   }
//   stat->TinyPivots = 0;
//   stat->RefineSteps = 0;
//   stat->expansions = 0;
// }

// __host__ __device__ void set_default_options(superlu_options_t *options) {
//   options->Fact = DOFACT;
//   options->Equil = YES;
//   options->ColPerm = COLAMD;
//   options->Trans = NOTRANS;
//   options->IterRefine = NOREFINE;
//   options->DiagPivotThresh = 1.0;
//   options->SymmetricMode = NO;
//   options->PivotGrowth = NO;
//   options->ConditionNumber = NO;
//   options->PrintStat = YES;
// }

// __host__ __device__ void dCreate_CompCol_Matrix(SuperMatrix *A, int m, int n,
//                                                 int nnz, double *nzval,
//                                                 int *rowind, int *colptr,
//                                                 Stype_t stype, Dtype_t dtype,
//                                                 Mtype_t mtype) {
//   NCformat *Astore;
//   A->Stype = stype;
//   A->Dtype = dtype;
//   A->Mtype = mtype;
//   A->nrow = m;
//   A->ncol = n;
//   A->Store = (void *)malloc(sizeof(NCformat));
//   Astore = (NCformat *)A->Store;
//   Astore->nnz = nnz;
//   Astore->nzval = nzval;
//   Astore->rowind = rowind;
//   Astore->colptr = colptr;
// }

// __host__ __device__ void dCreate_Dense_Matrix(SuperMatrix *X, int m, int n,
//                                               double *x, int ldx, Stype_t
//                                               stype, Dtype_t dtype, Mtype_t
//                                               mtype) {
//   DNformat *Xstore;
//   X->Stype = stype;
//   X->Dtype = dtype;
//   X->Mtype = mtype;
//   X->nrow = m;
//   X->ncol = n;
//   X->Store = (void *)malloc(sizeof(DNformat));
//   Xstore = (DNformat *)X->Store;
//   Xstore->lda = ldx;
//   Xstore->nzval = (double *)x;
// }

class Triplet {
public:
  __host__ __device__ Triplet() : m_row(0), m_col(0), m_value(0) {}

  __host__ __device__ Triplet(int i, int j, float v)
      : m_row(i), m_col(j), m_value(v) {}

  __host__ __device__ const int &row() { return m_row; }
  __host__ __device__ const int &col() { return m_col; }
  __host__ __device__ const float &value() { return m_value; }

protected:
  int m_row, m_col;
  float m_value;
};

typedef Triplet sparse_element;

__global__ void solve_sparse_linear_alt(sparse_element *ABCD_matrix,
                                        float *E_matrix, int non_zero_elements,
                                        int m, int n) {
  // superlu_options_t options;
  // SuperLUStat_t stat;
  // SuperMatrix A, B;
  // double *a = (double *)malloc(sizeof(double) * non_zero_elements);
  // a[0] = 1.2;
  // a[1] = 0.9;
  // int *asub = (int *)malloc(sizeof(int) * non_zero_elements);
  // asub[0] = 0;
  // asub[1] = 1;
  // int *xa = (int *)malloc(sizeof(int) * non_zero_elements);
  // xa[0] = 1;
  // xa[1] = 2;
  // dCreate_CompCol_Matrix(&A, m, n, non_zero_elements, a, asub, xa, SLU_NC,
  //                        SLU_D, SLU_GE);
  // int nrhs = 1;
  // double *rhs = (double *)malloc(m * nrhs);
  // rhs[0] = 0.5;
  // dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);
  // set_default_options(&options);
  // options.ColPerm = NATURAL;
  // StatInit(&stat);
  // cudaSafeCall(Eigen::SparseMatrix<float> ABCD(2 * m * n, 2 * m * n));
  // // ABCD.setFromTriplets(&ABCD_matrix[0], &ABCD_matrix[non_zero_elements]);
  // // ABCD.makeCompressed();
  // Eigen::Map<Eigen::VectorXf> E(E_matrix, 2 * m * n);
}

__global__ void gen_ABE_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor,
    float *V_WL_accessor, float *V_BL_accessor, int m, int n, float R_source,
    float R_line, sparse_element *ABCD_matrix, float *E_matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  if (i < m && j < n) {
    int index = (i * n + j) * 5;
    // A matrix
    if (j == 0) {
      E_matrix[i * n] = V_WL_accessor[i] / R_source; // E matrix (partial)
      ABCD_matrix[index] = sparse_element(i * n, i * n,
                                          conductance_matrix_accessor[i][0] +
                                              1.0f / R_source + 1.0f / R_line);
    } else {
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    ABCD_matrix[index] =
        sparse_element(i * n + j, i * n + j,
                       conductance_matrix_accessor[i][j] + 2.0f / R_line);
    index++;
    if (j < n - 1) {
      ABCD_matrix[index] =
          sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line);
      index++;
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line);
    } else {
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j,
                         conductance_matrix_accessor[i][j] + 1.0 / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // B matrix
    ABCD_matrix[index] = sparse_element(i * n + j, i * n + j + (m * n),
                                        -conductance_matrix_accessor[i][j]);
  }
}

__global__ void gen_CDE_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor,
    float *V_WL_accessor, float *V_BL_accessor, int m, int n, float R_source,
    float R_line, sparse_element *ABCD_matrix, float *E_matrix) {
  int j = threadIdx.x + blockIdx.x * blockDim.x; // for (int j = 0; j < n; j++)
  int i = threadIdx.y + blockIdx.y * blockDim.y; // for (int i = 0; i < m; i++)
  if (j < n && i < m) {
    int index = (5 * m * n) + ((j * m + i) * 4);
    // D matrix
    if (i == 0) {
      E_matrix[m * n + (j + 1) * m - 1] =
          -V_BL_accessor[j] / R_source; // E matrix (partial)
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m), m * n + j,
                         -1.0f / R_line - conductance_matrix_accessor[0][j]);
      index++;
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m), m * n + j + n, 1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    } else if (i < m - 1) {
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 1.0f / R_line);
      index++;
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
                         -conductance_matrix_accessor[i][j] - 2.0f / R_line);
    } else {
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 2)) + j, 1 / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
          -1.0f / R_source - conductance_matrix_accessor[m - 1][j] -
              1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // C matrix
    ABCD_matrix[index] = sparse_element(j * m + i + (m * n), n * i + j,
                                        conductance_matrix_accessor[i][j]);
  }
}

__global__ void
construct_V_applied(torch::PackedTensorAccessor32<float, 2> V_applied_accessor,
                    float *V_accessor, int m, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  if (i < m && j < n) {
    V_applied_accessor[i][j] =
        V_accessor[n * i + j] - V_accessor[m * n + n * i + j];
  }
}

at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
                         at::Tensor V_BL, float R_source, float R_line) {
  assert(at::cuda::is_available());
  conductance_matrix = conductance_matrix.to(torch::Device("cuda:0"));
  V_WL = V_WL.to(torch::Device("cuda:0"));
  V_BL = V_BL.to(torch::Device("cuda:0"));
  int m = conductance_matrix.sizes()[0];
  int n = conductance_matrix.sizes()[1];
  torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor =
      conductance_matrix.packed_accessor32<float, 2>();
  float *V_WL_accessor = V_WL.data_ptr<float>();
  float *V_BL_accessor = V_BL.data_ptr<float>();
  int non_zero_elements =
      (5 * m * n) +
      (4 * m * n); // Uncompressed (with padding for CUDA execution).
  // When compressed, contains 8 * m * n - 2 * m - 2 * n unique values.
  sparse_element *ABCD_matrix;
  sparse_element *ABCD_matrix_host =
      (sparse_element *)malloc(sizeof(sparse_element) * non_zero_elements);
  cudaMalloc(&ABCD_matrix, sizeof(sparse_element) * non_zero_elements);
  float *E_matrix;
  cudaMalloc(&E_matrix, sizeof(float) * 2 * m * n);
  cudaMemset(E_matrix, 0, sizeof(float) * 2 * m * n);
  float *E_matrix_host = (float *)malloc(sizeof(float) * 2 * m * n);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads = prop.maxThreadsDim[0];
  dim3 grid;
  dim3 block;
  if (m * n > max_threads) {
    int n_grid = ceil_int_div(m * n, max_threads);
    grid = dim3(n_grid, n_grid, 1);
    block = dim3(ceil_int_div(m, n_grid), ceil_int_div(n, n_grid), 1);
  } else {
    grid = dim3(1, 1, 1);
    block = dim3(m, n, 1);
  }
  gen_ABE_kernel<<<grid, block>>>(conductance_matrix_accessor, V_WL_accessor,
                                  V_BL_accessor, m, n, R_source, R_line,
                                  ABCD_matrix, E_matrix);
  gen_CDE_kernel<<<grid, block>>>(conductance_matrix_accessor, V_WL_accessor,
                                  V_BL_accessor, m, n, R_source, R_line,
                                  ABCD_matrix, E_matrix);
  cudaSafeCall(cudaDeviceSynchronize());
  // Eigen::SparseMatrix<float> ABCD(2 * m * n, 2 * m * n);
  // cudaMemcpy(ABCD_matrix_host, ABCD_matrix,
  //            sizeof(sparse_element) * non_zero_elements,
  //            cudaMemcpyDeviceToHost);
  // ABCD.setFromTriplets(&ABCD_matrix_host[0],
  //                      &ABCD_matrix_host[non_zero_elements]);
  // ABCD.makeCompressed();
  // cudaMemcpy(E_matrix_host, E_matrix, sizeof(float) * 2 * m * n,
  //            cudaMemcpyDeviceToHost);
  // Eigen::Map<Eigen::VectorXf> E(E_matrix_host, 2 * m * n);
  // Eigen::VectorXf V = solve_sparse_linear(ABCD, E);
  solve_sparse_linear_alt<<<(1, 1, 1), (1, 1, 1)>>>(ABCD_matrix, E_matrix,
                                                    non_zero_elements, m, n);
  cudaSafeCall(cudaDeviceSynchronize());
  cudaError_t c_ret = cudaGetLastError();
  if (c_ret) {
    std::cout << "Error: " << cudaGetErrorString(c_ret) << "-->";
  }
  at::Tensor V_applied_tensor =
      at::zeros({m, n}, torch::TensorOptions().device(torch::kCUDA, 0));
  // torch::PackedTensorAccessor32<float, 2> V_applied_accessor =
  //     V_applied_tensor.packed_accessor32<float, 2>();
  // float *V_accessor;
  // cudaMalloc(&V_accessor, sizeof(float) * V.size());
  // cudaMemcpy(V_accessor, V.data(), sizeof(float) * V.size(),
  //            cudaMemcpyHostToDevice);
  // construct_V_applied<<<grid, block>>>(V_applied_accessor, V_accessor, m, n);
  // cudaSafeCall(cudaDeviceSynchronize());
  // cudaSafeCall(cudaFree(ABCD_matrix));
  // cudaSafeCall(cudaFree(E_matrix));
  // cudaSafeCall(cudaFree(V_accessor));
  // cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  return at::sum(at::mul(V_applied_tensor, conductance_matrix), 0);
}