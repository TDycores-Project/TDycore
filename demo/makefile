TARGETS = mpfao sample

ALL: $(TARGETS)
clean::
	-@$(RM) $(TARGETS)

topdir := $(shell cd .. && pwd)
TDYCORE_DIR ?= $(topdir)
include $(TDYCORE_DIR)/lib/tdycore/conf/variables
include $(TDYCORE_DIR)/lib/tdycore/conf/rules

mpfao: mpfao.o chkopts
	$(CLINKER) -o $@ $< $(TDYCORE_LIB)
	$(RM) -f $<

sample: sample.o chkopts
	$(CLINKER) -o $@ $< $(TDYCORE_LIB)
	$(RM) -f $<

