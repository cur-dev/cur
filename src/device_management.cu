#include <cuda_runtime.h>
#include <Rinternals.h>


extern "C" SEXP R_cudaGetDeviceCount()
{
  SEXP ret;
  cudaError_t check;
  
  PROTECT(ret = allocVector(INTSXP, 1));
  check = cudaGetDeviceCount(INTEGER(ret));
  UNPROTECT(1);
  return ret;
}



extern "C" SEXP R_cudaGetDevice()
{
  SEXP ret;
  cudaError_t check;
  
  PROTECT(ret = allocVector(INTSXP, 1));
  check = cudaGetDevice(INTEGER(ret));
  UNPROTECT(1);
  return ret;
}



extern "C" SEXP R_cudaSetDevice(SEXP device)
{
  cudaError_t check;
  
  check = cudaSetDevice(INTEGER(device)[0]);
  return R_NilValue;
}



extern "C" SEXP R_cudaDeviceReset()
{
  cudaError_t check;
  
  check = cudaDeviceReset();
  return R_NilValue;
}
