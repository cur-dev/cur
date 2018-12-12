DEVICE_TO_HOST = 1L
HOST_TO_DEVICE = 2L
DEVICE_TO_DEVICE = 3L

TYPE_INT = 1L
TYPE_FLOAT = 2L
TYPE_DOUBLE = 3L

global_str2int = function(str)
{
  if (str == "devicetohost")
    DEVICE_TO_HOST
  else if (str == "hosttodevice")
    HOST_TO_DEVICE
  else if (str == "devicetodevice")
    DEVICE_TO_DEVICE
  
  else if (str == "int")
    TYPE_INT
  else if (str == "float")
    TYPE_FLOAT
  else if (str == "double")
    TYPE_DOUBLE
  
  else
    stop("internal error: please report this to the developers")
}
