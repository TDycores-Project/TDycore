TARGETS = mpfao

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
	-@$(RM) $(TARGETS) *.gcov *.regression *.old *.stdout *.bin *.bin.info

topdir := $(shell cd ../.. && pwd)
TDYCORE_DIR ?= $(topdir)
include $(TDYCORE_DIR)/lib/tdycore/conf/variables
include $(TDYCORE_DIR)/lib/tdycore/conf/rules

build:
	@cd ../; make

mpfao: mpfao.o chkopts
	$(CLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

streamline: mpfao_streamline.o chkopts
	$(CLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

