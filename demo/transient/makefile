TARGETS = transient transient_mpfaof90 transient_snes_mpfaof90

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
	-@$(RM) $(TARGETS) *.gcov *.regression *.old *.stdout *xmf *.vtk *.bin*

topdir := $(shell cd ../.. && pwd)
TDYCORE_DIR ?= $(topdir)
include $(TDYCORE_DIR)/lib/tdycore/conf/variables
include $(TDYCORE_DIR)/lib/tdycore/conf/rules

build:
	@cd ../; make

transient: transient.o
	$(CLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

transient_mpfaof90: transient_mpfaof90.o
	$(FLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

transient_snes_mpfaof90: transient_snes_mpfaof90.o
	$(FLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

