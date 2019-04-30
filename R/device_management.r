#' CUDA Device Management
#' 
#' CUDA device management utilities.
#' 
#' @details
#' \code{cudaDeviceReset()} Destroys all allocations and resets state on the
#' current device in the current process.
#' 
#' \code{cudaGetDevice()} returns the device number (0, 1, ...) currently in use.
#' 
#' \code{cudaGetDeviceCount()} returns the number of available GPU's.
#' 
#' \code{cudaSetDevice()} sets the device to the supplied number (0, 1, ...).
#' 
#' @param device
#' A non-negative integer corresponding to the GPU you want to use.
#' 
#' @references NVIDIA CUDA Runtime API
#' \url{https://docs.nvidia.com/cuda/cuda-runtime-api/index.html}
#' 
#' @name device_management
#' @rdname device_management
NULL



#' @useDynLib cur R_cudaDeviceReset
#' @rdname device_management
#' @export
cudaDeviceReset = function()
{
  .Call(R_cudaDeviceReset)
  invisible()
}



#' @useDynLib cur R_cudaGetDevice
#' @rdname device_management
#' @export
cudaGetDevice = function()
{
  .Call(R_cudaGetDevice)
}



#' @useDynLib cur R_cudaGetDeviceCount
#' @rdname device_management
#' @export
cudaGetDeviceCount = function()
{
  .Call(R_cudaGetDeviceCount)
}



#' @useDynLib cur R_cudaSetDevice
#' @rdname device_management
#' @export
cudaSetDevice = function(device)
{
  .Call(R_cudaSetDevice, as.integer(device))
  invisible()
}
