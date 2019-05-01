#ifndef CUR_CHECK_HH
#define CUR_CHECK_HH


#include <cuda_runtime.h>
#include <Rinternals.h>

#define CHECK_CUDA(call) {cudaError_t check = call; check_cuda_ret(check);}

static inline void check_cuda_ret(cudaError_t check)
{
  if (check != cudaSuccess)
    error("%s", cudaGetErrorString(check));
}


#endif
