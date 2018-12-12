#include <cuda_runtime.h>
#include <Rinternals.h>


#define COPY_TO_HOST 1
#define COPY_TO_DEVICE 2

#define TYPE_INT 1
#define TYPE_FLOAT 2
#define TYPE_DOUBLE 3

#define newRptr(ptr,Rptr,fin) PROTECT(Rptr = R_MakeExternalPtr(ptr, R_NilValue, R_NilValue));R_RegisterCFinalizerEx(Rptr, fin, TRUE)
#define getRptr(ptr) R_ExternalPtrAddr(ptr)


static inline void cuda_object_finalizer(SEXP Rptr)
{
  cudaError_t check;
  void *x = getRptr(Rptr);
  
  if (x == NULL)
    return;
  
  check = cudaFree(x);
  R_ClearExternalPtr(Rptr);
}



extern "C" SEXP R_cudaMalloc(SEXP n, SEXP size)
{
  cudaError_t check;
  SEXP ret;
  void *x;
  
  size_t len = (size_t) REAL(n)[0] * INTEGER(size)[0];
  check = cudaMalloc(&x, len);
  newRptr(x, ret, cuda_object_finalizer);
  
  UNPROTECT(1);
  return ret;
}



extern "C" SEXP R_cudaFree(SEXP x_ptr)
{
  cudaError_t check;
  void *x = getRptr(x_ptr);
  
  check = cudaFree(x);
  return R_NilValue;
}



extern "C" SEXP R_cudaMemGetInfo()
{
  SEXP ret, ret_names;
  SEXP free, total;
  cudaError_t check;
  size_t mem_free, mem_total;
  
  PROTECT(ret = allocVector(VECSXP, 2));
  PROTECT(ret_names = allocVector(STRSXP, 2));
  
  PROTECT(free = allocVector(REALSXP, 1));
  PROTECT(total = allocVector(REALSXP, 1));
  
  check = cudaMemGetInfo(&mem_free, &mem_total);
  
  REAL(free)[0] = (double) mem_free;
  REAL(total)[0] = (double) mem_total;
  
  SET_VECTOR_ELT(ret, 0, free);
  SET_VECTOR_ELT(ret, 1, total);
  SET_STRING_ELT(ret_names, 0, mkChar("free"));
  SET_STRING_ELT(ret_names, 1, mkChar("total"));
  setAttrib(ret, R_NamesSymbol, ret_names);
  
  UNPROTECT(4);
  return ret;
}



#define SET_ROBJ_PTR(ptr, R_ptr) \
  if (TYPEOF(R_ptr) == INTSXP){ \
    ptr = (void*) INTEGER(R_ptr); \
  } else if (TYPEOF(R_ptr) == REALSXP){ \
    ptr = (void*) REAL(R_ptr); \
  }

extern "C" SEXP R_cudaMemcpy(SEXP dst_, SEXP src_, SEXP count, SEXP size, SEXP kind_)
{
  cudaError_t check;
  void *dst;
  void *src;
  
  int kind = INTEGER(kind_)[0];
  size_t len = (size_t) REAL(count)[0] * INTEGER(size)[0];
  
  if (kind == COPY_TO_HOST)
  {
    SET_ROBJ_PTR(dst, dst_);
    src = getRptr(src_);
    check = cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost);
  }
  else if (kind == COPY_TO_DEVICE)
  {
    dst = getRptr(dst_);
    SET_ROBJ_PTR(src, src_);
    check = cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice);
  }
  
  return R_NilValue;
}



extern "C" SEXP R_cudaMemset(SEXP x_ptr, SEXP value, SEXP count, SEXP size)
{
  cudaError_t check;
  
  void *x = getRptr(x_ptr);
  size_t len = (size_t) REAL(count)[0] * INTEGER(size)[0];
  check = cudaMemset(x, INTEGER(value)[0], len);
  
  return R_NilValue;
}
