#' CUDA Memory Management
#' 
#' CUDA memory management utilities.
#' 
#' 
#' 
#' 
#' @name memory_management
#' @rdname memory_management
NULL



#' @useDynLib cur R_cudaMalloc
#' @rdname memory_management
#' @export
cudaMalloc = function(count, size)
{
  count = as.double(count)
  
  size = match.arg(tolower(size), c("int", "float", "double"))
  size = global_str2int(size)
  
  ret = .Call(R_cudaMalloc, count, size)
  class(ret) = "cuda_device_memory"
  
  ret
}



#' @useDynLib cur R_cudaFree
#' @rdname memory_management
#' @export
cudaFree = function(dev_ptr)
{
  .Call(R_cudaFree, dev_ptr)
  invisible()
}



#' @useDynLib cur R_cudaMemGetInfo
#' @rdname memory_management
#' @export
cudaMemGetInfo = function()
{
  .Call(R_cudaMemGetInfo)
}



#' @useDynLib cur R_cudaMemcpy
#' @rdname memory_management
#' @export
cudaMemcpy = function(dst, src, count, size, kind)
{
  count = as.double(count)
  
  size = match.arg(tolower(size), c("int", "float", "double"))
  size = global_str2int(size)
  
  kind = match.arg(tolower(kind), c("devicetohost", "hosttodevice"))
  kind = global_str2int(kind)
  
  if (kind == COPY_TO_HOST)
  {
    if (!inherits(src, "cuda_device_memory"))
      stop("copying device to host: 'src' array must be CUDA allocated memory")
    
    if (float::is.float(src))
      dst = dst@Data
    else if (!is.double(dst) && !is.integer(dst))
      stop("copying device to host: 'dst' must be an int, float, or double vector")
  }
  else # kind == COPY_TO_DEVICE
  {
    if (!inherits(dst, "cuda_device_memory"))
      stop("copying host to device: 'dst' array must be CUDA allocated memory")
    
    if (float::is.float(src))
      src = src@Data
    else if (!is.double(src) && !is.integer(src))
      stop("copying host to device: 'src' must be an int, float, or double vector")
  }
  
  
  .Call(R_cudaMemcpy, dst, src, count, size, kind)
  invisible()
}



#' @useDynLib cur R_cudaMemset
#' @rdname memory_management
#' @export
cudaMemset = function(dev_ptr, value, count, size)
{
  count = as.double(count)
  
  size = match.arg(tolower(size), c("int", "float", "double"))
  size = global_str2int(size)
  
  .Call(R_cudaMemset, dev_ptr, as.integer(value), count, size)
  invisible()
}
