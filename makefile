ALL: all
LOCDIR = .
DIRS   = include src docs test demo

ifdef codecov
  MYFLAGS += -fprofile-arcs -ftest-coverage
  LIBS    += -lgcov
endif

# These flags are supplemental to the PETSc flags
CFLAGS   = ${LIBS}
FFLAGS   = ${LIBS}
CPPFLAGS = ${MYFLAGS}
FPPFLAGS = ${MYFLAGS}

TDYCORE_DIR ?= $(CURDIR)
include ${TDYCORE_DIR}/lib/tdycore/conf/variables
include ${TDYCORE_DIR}/lib/tdycore/conf/rules
include ${TDYCORE_DIR}/lib/tdycore/conf/test

all:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	${OMAKE} all-gmake  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}; \
	elif [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
	${OMAKE} all-cmake  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}; else \
	${OMAKE} all-legacy PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}; fi;
.PHONY: all


${TDYCORE_DIR}/${PETSC_ARCH}/include:
	@${MKDIR} ${TDYCORE_DIR}/${PETSC_ARCH}/include
${TDYCORE_DIR}/${PETSC_ARCH}/lib:
	@${MKDIR} ${TDYCORE_DIR}/${PETSC_ARCH}/lib
${TDYCORE_DIR}/${PETSC_ARCH}/log:
	@${MKDIR} ${TDYCORE_DIR}/${PETSC_ARCH}/log
arch-tree: ${TDYCORE_DIR}/${PETSC_ARCH}/include \
           ${TDYCORE_DIR}/${PETSC_ARCH}/lib \
	   ${TDYCORE_DIR}/${PETSC_ARCH}/log
.PHONY: arch-tree


#
# GNU Make build
#
ifndef MAKE_IS_GNUMAKE
MAKE_IS_GNUMAKE = $(if $(findstring GNU Make,$(shell $(OMAKE) --version 2>/dev/null)),1,)
endif
ifdef OMAKE_PRINTDIR
GMAKE = ${OMAKE_PRINTDIR}
else
GMAKE = ${OMAKE}
endif
gmake-build:
	@cd ${TDYCORE_DIR} && ${GMAKE} -f gmakefile -j ${MAKE_NP}
gmake-clean:
	@cd ${TDYCORE_DIR} && ${GMAKE} -f gmakefile clean
all-gmake: chk_petsc_dir chk_tdycore_dir arch-tree
	-@echo "=================================================="
	-@echo "Building TDycore (GNU Make - ${MAKE_NP} build jobs)"
	-@echo "Using TDYCORE_DIR=${TDYCORE_DIR}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "=================================================="
	@${GMAKE} gmake-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR} CFLAGS="${CFLAGS}" FFLAGS="${FFLAGS}" CPPFLAGS="${CPPFLAGS}" FPPFLAGS="${FPPFLAGS}"  2>&1 | tee ./${PETSC_ARCH}/log/make.log
	-@echo "=================================================="
.PHONY: gmake-build gmake-clean all-gmake


#
# CMake build
#
ifeq (${PETSC_LANGUAGE},CXXONLY)
cmake_cc_clang=-DPETSC_CLANGUAGE_Cxx:STRING='YES'
cmake_cc_path =-DCMAKE_CXX_COMPILER:FILEPATH=${CXX}
cmake_cc_flags=-DCMAKE_CXX_FLAGS:STRING='${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'
else
cmake_cc_clang=-DPETSC_CLANGUAGE_Cxx:STRING='NO'
cmake_cc_path =-DCMAKE_C_COMPILER:FILEPATH=${CC}
cmake_cc_flags=-DCMAKE_C_FLAGS:STRING='${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'
endif
ifneq (${FC},)
cmake_fc_path =-DCMAKE_Fortran_COMPILER:FILEPATH=${FC}
endif
ifneq (${FC_FLAGS},)
cmake_fc_flags=-DCMAKE_Fortran_FLAGS:STRING='${FC_FLAGS} ${FFLAGS} ${FCPPFLAGS}'
endif
cmake_cc=${cmake_cc_path} ${cmake_cc_flags} ${cmake_cc_clang}
cmake_fc=${cmake_fc_path} ${cmake_fc_flags}
${TDYCORE_DIR}/${PETSC_ARCH}/CMakeCache.txt: CMakeLists.txt
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/CMakeCache.txt
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/CMakeFiles
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/Makefile
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/cmake_install.cmake
	@${MKDIR} ${TDYCORE_DIR}/${PETSC_ARCH}
	@cd ${TDYCORE_DIR}/${PETSC_ARCH} && ${CMAKE} ${TDYCORE_DIR} ${cmake_cc} ${cmake_fc}
cmake-boot: ${TDYCORE_DIR}/${PETSC_ARCH}/CMakeCache.txt
cmake-down:
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/CMakeCache.txt
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/CMakeFiles
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/Makefile
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/cmake_install.cmake
cmake-build: cmake-boot
	@cd ${TDYCORE_DIR}/${PETSC_ARCH} && ${OMAKE} -j ${MAKE_NP}
	-@if [ "${DSYMUTIL}" != "true" -a -f ${INSTALL_LIB_DIR}/libtdycore.${SL_LINKER_SUFFIX} ]; then \
        ${DSYMUTIL} ${INSTALL_LIB_DIR}/libtdycore.${SL_LINKER_SUFFIX}; fi
cmake-install:
	@cd ${TDYCORE_DIR}/${PETSC_ARCH} && ${OMAKE} install
cmake-clean:
	@if [ -f ${TDYCORE_DIR}/${PETSC_ARCH}/Makefile ]; then \
	cd ${TDYCORE_DIR}/${PETSC_ARCH} && ${OMAKE} clean; fi;
all-cmake: chk_petsc_dir chk_tdycore_dir arch-tree
	-@echo "=================================================="
	-@echo "Building TDycore (CMake - ${MAKE_NP} build jobs)"
	-@echo "Using TDYCORE_DIR=${TDYCORE_DIR}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "=================================================="
	@${OMAKE} cmake-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR} CFLAGS="${CFLAGS}" FFLAGS="${FFLAGS}" CPPFLAGS="${CPPFLAGS}" FPPFLAGS="${FPPFLAGS}" 2>&1 | tee ./${PETSC_ARCH}/log/make.log
	-@echo "=================================================="
.PHONY: cmake-boot cmake-down cmake-build cmake-clean all-cmake


#
# Legacy build
#
legacy-build: arch-tree deletelibs deletemods build
legacy-clean: deletemods deletelibs
	-@${OMAKE} tree ACTION=clean PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}
all-legacy: chk_petsc_dir chk_tdycore_dir arch-tree
	-@echo "=================================================="
	-@echo "Building TDycore (legacy build)"
	-@echo "Using TDYCORE_DIR=${TDYCORE_DIR}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "=================================================="
	-@echo "Beginning to build TDycore library"
	@${OMAKE} legacy-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR} 2>&1 | tee ./${PETSC_ARCH}/log/make.log
	-@echo "Completed building TDycore library"
	-@echo "=================================================="
.PHONY: legacy-build legacy-clean all-legacy

#
# Check if PETSC_DIR variable specified is valid
#
chk_petsc_dir:
	@if [ ! -f ${PETSC_DIR}/include/petsc.h ]; then \
	  echo "Incorrect PETSC_DIR specified: ${PETSC_DIR}"; \
	  echo "Aborting build"; \
	  false; fi
.PHONY: chk_petsc_dir

#
# Check if TDYCORE_DIR variable specified is valid
#
chk_tdycore_dir:
	@if [ ! -f ${TDYCORE_DIR}/include/tdycore.h ]; then \
	  echo "Incorrect TDYCORE_DIR specified: ${TDYCORE_DIR}"; \
	  echo "Aborting build"; \
	  false; fi
.PHONY: chk_tdycore_dir

#
# Build the TDycore library
#
build: compile ranlib shlibs
compile:
	-@${OMAKE} tree ACTION=libfast PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}
	-@${MV} -f ${TDYCORE_DIR}/src/tdycore*.mod ${TDYCORE_DIR}/${PETSC_ARCH}/include
ranlib:
	-@echo "building libtdycore.${AR_LIB_SUFFIX}"
	-@${RANLIB} ${TDYCORE_LIB_DIR}/*.${AR_LIB_SUFFIX} > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
shlibs:
	-@${OMAKE} shared_nomesg PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR} \
		   | (${GREP} -vE "making shared libraries in" || true) \
		   | (${GREP} -vE "==========================" || true)
	-@if [ "${DSYMUTIL}" != "true" ]; then \
        ${DSYMUTIL} ${INSTALL_LIB_DIR}/libtdycore.${SL_LINKER_SUFFIX}; fi
.PHONY: build compile ranlib shlibs

# Delete TDycore library
deletelogs:
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/log/*.log
deletemods:
	-@${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/include/tdycore*.mod
deletestaticlibs:
	-@${RM} -r ${TDYCORE_LIB_DIR}/libtdycore*.${AR_LIB_SUFFIX}
deletesharedlibs:
	-@${RM} -r ${TDYCORE_LIB_DIR}/libtdycore*.${SL_LINKER_SUFFIX}*
deletelibs: deletestaticlibs deletesharedlibs
.PHONY: deletelogs deletemods deletestaticlibs deletesharedlibs deletelibs


# Clean up build
clean:: allclean
allclean:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	${OMAKE} gmake-clean  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}; \
	elif [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
	${OMAKE} cmake-clean  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}; else \
	${OMAKE} legacy-clean PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}; fi;
distclean: chk_tdycore_dir
	@echo "*** Deleting all build files ***"
	-${RM} -r ${TDYCORE_DIR}/${PETSC_ARCH}/
.PHONY: clean allclean distclean


# Run test examples
testexamples:
	-@echo "=================================================="
	-@echo "Beginning to compile and run test examples"
	-@echo "=================================================="
	-@${OMAKE} tree ACTION=testexamples_C PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}
	-@echo "Completed compiling and running test examples"
	-@echo "=================================================="
.PHONY: testexamples


# Test build
check: test
test:
	-@${OMAKE} test-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR} 2>&1 | tee ./${PETSC_ARCH}/log/test.log
test-build:
	-@echo "Running test to verify correct installation"
	-@echo "Using TDYCORE_DIR=${TDYCORE_DIR}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	@cd test; ${OMAKE} clean       PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}
	@cd test; ${OMAKE} test-build  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}
	@cd test; ${OMAKE} clean       PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TDYCORE_DIR=${TDYCORE_DIR}
	-@echo "Completed test"
.PHONY: check test test-build


#
# Documentation
#
SRCDIR=${TDYCORE_DIR}/src
DOCDIR=${TDYCORE_DIR}/docs/html
doc:
	@if [ ! -d ${DOCDIR} ]; then ${MKDIR} ${DOCDIR}; fi
	-@${RM} ${DOCDIR}/*.html
	@${PETSC_DIR}/${PETSC_ARCH}/bin/doctext -mpath ${DOCDIR} -html ${SRCDIR}/*.c
	@echo '<TITLE>TDycore Documentation</TITLE>' > ${DOCDIR}/index.html
	@echo '<H1>TDycore Documentation</H1>' >> ${DOCDIR}/index.html
	@echo '<MENU>' >> ${DOCDIR}/index.html
	@ls -1 ${DOCDIR} | grep .html | grep -v index.html | sed -e 's%^\(.*\).html$$%<LI><A HREF="\1.html">\1</A>%g' >> ${DOCDIR}/index.html
	@echo '</MENU>' >> ${DOCDIR}/index.html
deletedoc:
	-@${RM} ${DOCDIR}/*.html
.PHONY: doc deletedoc


#
# TAGS Generation
#
alletags:
	-@${PYTHON} ${PETSC_DIR}/bin/maint/generateetags.py
deleteetags:
	-@${RM} CTAGS TAGS
.PHONY: alletags deleteetags
