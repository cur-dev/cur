#ifndef CUR_H_
#define CUR_H_

#include <Rinternals.h>

#define CUR_TYPE_INT 1
#define CUR_TYPE_FLOAT 2
#define CUR_TYPE_DOUBLE 3

#define CUR_ARRTYPE_VECTOR 1L
#define CUR_ARRTYPE_MATRIX 2L

#define CUR_ARRTYPE(x) (INTEGER(VECTOR_ELT(x, 0))[0])
#define CUR_TYPE(x) (INTEGER(VECTOR_ELT(x, 0))[1])
#define CUR_NROWS(x) (REAL(VECTOR_ELT(x, 1))[0])
#define CUR_NCOLS(x) (REAL(VECTOR_ELT(x, 1))[1])
#define CUR_LENGTH(x) (CUR_NROWS(x)*CUR_NCOLS(x))
#define CUR_DATA(x) (R_ExternalPtrAddr(VECTOR_ELT(x, 2)))


#endif
