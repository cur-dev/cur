check_margin = function(m)
{
  if (is.null(m) || !is.numeric(m) || length(m) != 1 || is.na(m) || m < 0)
    stop("'nrow' and 'ncol' should each be ")
  
  as.double(m)
}

cur_arr = function(arrtype, datatype, nr, nc, ptr)
{
  nr = check_margin(nr)
  nc = check_margin(nc)
  
  list(c(arrtype, datatype), c(nr, nc), ptr)
}



#' @export
int_cuda = function(nrow=0L, ncol)
{
  if (missing(ncol))
    ncol = 1
  
  x = .Call(R_cudaMalloc, length, TYPE_INT)
  ret = cur_arr(arrtype=ARRTYPE_VECTOR, datatype=TYPE_INT, nr=nrow, nc=ncol, ptr=x)
  class(ret) = "cuda_array"
  ret
}

#' @export
float_cuda = function(nrow=0L, ncol)
{
  if (missing(ncol))
    ncol = 1
  
  x = .Call(R_cudaMalloc, length, TYPE_FLOAT)
  ret = cur_arr(arrtype=ARRTYPE_VECTOR, datatype=TYPE_FLOAT, nr=nrow, nc=ncol, ptr=x)
  class(ret) = "cuda_array"
  ret
}

#' @export
double_cuda = function(nrow=0L, ncol)
{
  if (missing(ncol))
    ncol = 1
  
  x = .Call(R_cudaMalloc, length, TYPE_DOUBLE)
  ret = cur_arr(arrtype=ARRTYPE_VECTOR, datatype=TYPE_DOUBLE, nr=nrow, nc=ncol, ptr=x)
  class(ret) = "cuda_array"
  ret
}
