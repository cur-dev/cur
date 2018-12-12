COPY_TO_HOST = 1L
COPY_TO_DEVICE = 2L

TYPE_INT = 1L
TYPE_FLOAT = 2L
TYPE_DOUBLE = 3L

global_str2int = function(str)
{
  if (str == tolower("cudamemcpydevicetohost"))
    COPY_TO_HOST
  else if (str == tolower("cudamemcpyhosttodevice"))
    COPY_TO_DEVICE
  
  else if (str == "int")
    TYPE_INT
  else if (str == "float")
    TYPE_FLOAT
  else if (str == "double")
    TYPE_DOUBLE
}