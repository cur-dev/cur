/* Automatically generated. Do not edit by hand. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

// device_management.cu
extern SEXP R_cudaGetDeviceCount();
extern SEXP R_cudaGetDevice();
extern SEXP R_cudaSetDevice(SEXP device);
extern SEXP R_cudaDeviceReset();
// memory_management.cu
extern SEXP R_cudaMalloc(SEXP n, SEXP size);
extern SEXP R_cudaFree(SEXP x_ptr);
extern SEXP R_cudaMemGetInfo();
extern SEXP R_cudaMemcpy(SEXP dst_ptr, SEXP src_ptr, SEXP count, SEXP size, SEXP kind_);
extern SEXP R_cudaMemset(SEXP dev_ptr, SEXP value, SEXP count, SEXP size);

static const R_CallMethodDef CallEntries[] = {
  {"R_cudaGetDeviceCount", (DL_FUNC) &R_cudaGetDeviceCount, 0},
  {"R_cudaGetDevice", (DL_FUNC) &R_cudaGetDevice, 0},
  {"R_cudaSetDevice", (DL_FUNC) &R_cudaSetDevice, 1},
  {"R_cudaDeviceReset", (DL_FUNC) &R_cudaDeviceReset, 0},
  
  {"R_cudaMalloc", (DL_FUNC) &R_cudaMalloc, 2},
  {"R_cudaFree", (DL_FUNC) &R_cudaFree, 1},
  {"R_cudaMemGetInfo", (DL_FUNC) &R_cudaMemGetInfo, 0},
  {"R_cudaMemcpy", (DL_FUNC) &R_cudaMemcpy, 5},
  {"R_cudaMemset", (DL_FUNC) &R_cudaMemset, 4},
  
  {NULL, NULL, 0}
};

void R_init_coop(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
