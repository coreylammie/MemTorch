// #pragma once
// #include <ST_TO_CC.cuh>

// /* C = A' */
// CUDA_CALLABLE_MEMBER
// cs *cs_transpose(const cs *A, csi values) {
//   cs *C = (cs *)malloc(sizeof(cs));
//   int non_zero_elements = A->nzmax;
//   C->m = A->n;
//   C->n = A->m;
//   C->nzmax = non_zero_elements;
//   C->nz = (csi)-1;
//   C->p = (csi *)malloc(sizeof(csi) * non_zero_elements);
//   C->i = (csi *)malloc(sizeof(csi) * non_zero_elements);
//   C->x = (double *)malloc(sizeof(double) * non_zero_elements);
//   memcpy(C->p, A->i, sizeof(csi) * non_zero_elements);
//   memcpy(C->i, A->p, sizeof(csi) * non_zero_elements);
//   memcpy(C->x, A->x, sizeof(double) * non_zero_elements);
//   st_to_cc(non_zero_elements, C->p, C->i, C->x);
//   return C;
// }
