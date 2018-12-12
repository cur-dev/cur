#' CUDA Device Management
#' 
#' CUDA device management utilities.
#' 
#' 
#' @name device_management
#' @rdname device_management
NULL



#' @useDynLib cur R_cudaGetDeviceCount
#' @rdname device_management
#' @export
cudaGetDeviceCount = function()
{
  .Call(R_cudaGetDeviceCount)
}



#' @useDynLib cur R_cudaGetDevice
#' @rdname device_management
#' @export
cudaGetDevice = function()
{
  .Call(R_cudaGetDevice)
}



#' @useDynLib cur R_cudaSetDevice
#' @rdname device_management
#' @export
cudaSetDevice = function(device)
{
  .Call(R_cudaSetDevice, as.integer(device))
  invisible()
}



#' @useDynLib cur R_cudaDeviceReset
#' @rdname device_management
#' @export
cudaDeviceReset = function()
{
  .Call(R_cudaDeviceReset)
  invisible()
}
