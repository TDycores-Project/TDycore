TARGETS = transient transient_mpfao transient_mpfaof90 transient_snes_mpfaof90

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

topdir := $(shell cd ../.. && pwd)
TDYCORE_DIR ?= $(topdir)
include $(TDYCORE_DIR)/lib/tdycore/conf/variables
include $(TDYCORE_DIR)/lib/tdycore/conf/rules

build:
	@cd ../; make

transient: transient.o chkopts
	$(CLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

transient_mpfao: transient_mpfao.o chkopts
	$(CLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

transient_th_mpfao: transient_th_mpfao.o chkopts
	$(CLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

transient_mpfaof90: transient_mpfaof90.o chkopts
	$(FLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

transient_snes_mpfaof90: transient_snes_mpfaof90.o chkopts
	$(FLINKER) -o $@ $< $(TDYCORE_LIB) $(LIBS)
	$(RM) -f $<

