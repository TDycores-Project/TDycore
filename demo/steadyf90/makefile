TARGETS = steadyf90

ifdef codecov
  MYFLAGS += -fprofile-arcs -ftest-coverage
  LIBS    += -lgcov
endif

# These flags are supplemental to the PETSc flags
CFLAGS   = ${LIBS}
FFLAGS   = ${LIBS}
CPPFLAGS = ${MYFLAGS}
ifndef CI # Internal compiler error in CI environment with GFortran 10.2
  FPPFLAGS = ${MYFLAGS}
endif

ALL: $(TARGETS)
clean::
	-@$(RM) $(TARGETS) *.gcov *.regression *.old *.stdout

topdir := $(shell cd ../../ && pwd)
TDYCORE_DIR ?= $(topdir)
include $(TDYCORE_DIR)/lib/tdycore/conf/variables
include $(TDYCORE_DIR)/lib/tdycore/conf/rules

build:
	@cd ../; make

steadyf90: steadyf90.o
	$(FLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<
