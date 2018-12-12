#' CUDA Memory Management
#' 
#' CUDA memory management utilities.
#' 
#' @details
#' \code{cudaMalloc()} allocates device memory and returns an external pointer.
#' This memory is managed by the R garbage collector.
#' 
#' \code{cudaFree()} Manually frees device memory. Does not destroy the R object
#' (so calling this will make the pointer managed by the R object invalid).
#' Use of this function will not cause a double free when the R object is gc'd.
#' 
#' \code{cudaMemGetInfo()} returns a list containing the number of free bytes
#' and the number of total bytes of memory on the current device.
#' 
#' \code{cudaMemcpy()} copies memory host-to-device, device-to-host, or
#' device-to-device.
#' 
#' \code{cudaMemset()} memset for device memory.
#' 
#' @param count
#' Number of elements (NOT BYTES).
#' @param size
#' A standin for \code{sizeof()}. Acceptable values are \code{"int"},
#' \code{"float"}, and \code{"double"}. Values are case insensitive.
#' @param kind
#' Description of the kind of copy. Acceptable values are \code{"DeviceToHost"},
#' \code{"HostToDevice"}, and \code{"DeviceToDevice"}. Values are case
#' insensitive.
#' @param dev_ptr
#' Pointer to device memory, i.e. an object returned from \code{cudaMalloc()}.
#' @param dst,src
#' Device/host vectors. \code{dst}-\code{src} should be one of device-host,
#' host-device, or device-device. Host vectors should be integer, float, or
#' double.
#' @param value
#' Integer value. Probably \code{0}.
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
