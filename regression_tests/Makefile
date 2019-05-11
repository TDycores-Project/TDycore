# Makefile for running standard regression tests
#
# include the make variables from PETSc so we can use PYTHON and do
# some conditional testing, e.g. test if unstructured mesh is available.
#
# If PYTHON defined from petsc is not correct, override it on the
# command line with: make PYTHON=python3.3 test

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


TEST_MANAGER = regression_tests.py

TEST_OPTIONS =

# make PERFORMANCE=true check
ifdef PERFORMANCE
	TEST_OPTIONS += --check-performance
endif

ifdef BACKTRACE
	TEST_OPTIONS += --backtrace
endif

ifdef TIMEOUT
	TEST_OPTIONS += --timeout $(TIMEOUT)
endif

ifneq ($(strip $(MPIEXEC)),)
	TEST_OPTIONS += --mpiexec $(MPIEXEC)
endif

#
# standard tests that are run to verify executable is built correctly
#

test : test-mpfao test-mpfaof90 test-steady test-transient

test-steady :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/steady $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/steady.cfg \
		--logfile_prefix steady

test-transient :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/transient $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/transient.cfg \
		--logfile_prefix transient

test-mpfao :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/mpfao $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/mpfao.cfg \
		--logfile_prefix mpfao

test-mpfaof90 :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/mpfaof90 $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/mpfaof90.cfg \
		--logfile_prefix mpfaof90

#
# Try to cleanup the regression test output.
#
# NOTE: do NOT add a "*.h5" glob. The .h5 extension is used for both
# input and output files, and a simple glob can not distinguish
# between them. Manually add rm commands for each file that needs to be
# removed.
#
clean-tests :
	-@find . -type f -name '*.testlog' -print0 | xargs -0 rm
	-@find . -type f -name '*.out' -print0 | xargs -0 rm

