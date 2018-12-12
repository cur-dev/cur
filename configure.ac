AC_INIT(DESCRIPTION)
AC_CONFIG_MACRO_DIR([tools/])

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
R_CPPFLAGS=`"${R_HOME}/bin/R" CMD config --cppflags`

AC_PROG_CC_C99
AX_CHECK_CUDA()

# Report
echo " "
echo "****************** Results of curand package configure ******************"
echo " "
echo "* R Report:"
echo "    >> R_CPPFLAGS=${R_CPPFLAGS}"
echo "* CUDA Report:"
echo "    >> NVCC=${NVCC}"
echo "    >> CUDA_CFLAGS=${CUDA_CFLAGS}"
echo "    >> CUDA_LDFLAGS=${CUDA_LDFLAGS}"
echo "*************************************************************************"
echo " "



dnl Start to substitute templates
AC_SUBST(NVCC)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(R_CPPFLAGS)
AC_OUTPUT(src/Makevars)
