TARGETS = build-steady build-transient build-richards build-th build-salinity

ifdef codecov
  MYFLAGS += -fprofile-arcs -ftest-coverage
  LIBS    += -lgcov
endif

# These flags are supplemental to the PETSc flags
CFLAGS   = ${LIBS}
FFLAGS   = ${LIBS}
CPPFLAGS = ${MYFLAGS}
FPPFLAGS = ${MYFLAGS}

ALL: $(TARGETS)
clean::
	-@$(RM) $(TARGETS) *.gcov *.regression *.old *.stdout

topdir := $(shell cd .. && pwd)
TDYCORE_DIR ?= $(topdir)
include $(TDYCORE_DIR)/lib/tdycore/conf/variables
include $(TDYCORE_DIR)/lib/tdycore/conf/rules

build:
	@cd ../; make

build-steady:
	@cd steady/; make

build-richards:
	@cd richards; make

build-th:
	@cd th; make

build-transient:
	@cd transient; make

build-salinity:
	@cd salinity; make

