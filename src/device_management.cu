#include <cuda_runtime.h>
#include <Rinternals.h>

#include "check.hh"


extern "C" SEXP R_cudaDeviceReset()
{
  CHECK_CUDA(cudaDeviceReset());
  return R_NilValue;
}



extern "C" SEXP R_cudaGetDevice()
{
  SEXP ret;
  PROTECT(ret = allocVector(INTSXP, 1));
  CHECK_CUDA(cudaGetDevice(INTEGER(ret)));
  UNPROTECT(1);
  return ret;
}



extern "C" SEXP R_cudaGetDeviceCount()
{
  SEXP ret;
  PROTECT(ret = allocVector(INTSXP, 1));
  CHECK_CUDA(cudaGetDeviceCount(INTEGER(ret)));
  UNPROTECT(1);
  return ret;
}



extern "C" SEXP R_cudaSetDevice(SEXP device)
{
  CHECK_CUDA(cudaSetDevice(INTEGER(device)[0]));
  return R_NilValue;
}
