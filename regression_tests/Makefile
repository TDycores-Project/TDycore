# Makefile for running standard regression tests
#
# include the make variables from PETSc so we can use PYTHON and do
# some conditional testing, e.g. test if unstructured mesh is available.
#
# If PYTHON defined from petsc is not correct, override it on the
# command line with: make PYTHON=python3.3 test

include ${PETSC_DIR}/lib/petsc/conf/variables

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
	TEST_OPTIONS += --mpiexec "$(MPIEXEC)"
endif

#
# standard tests that are run to verify executable is built correctly
#

test : test-mpfao test-steadyf90 test-steady test-transient test-transient-mpfao test-transient-mpfaof90 test-transient-snes-mpfaof90 test-richards

test-steady :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/steady/steady $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/steady/steady.cfg \
		--logfile_prefix steady

test-transient :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/transient/transient $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/transient/transient.cfg \
		--logfile_prefix transient

test-transient-mpfao :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/transient/transient_mpfao $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/transient/transient_mpfao.cfg \
		--logfile_prefix transient_mpfao

test-transient-mpfaof90 :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/transient/transient_mpfaof90 $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/transient/transient_mpfaof90.cfg \
		--logfile_prefix transient_mpfaof90

test-transient-snes-mpfaof90 :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/transient/transient_snes_mpfaof90 $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/transient/transient_snes_mpfaof90.cfg \
		--logfile_prefix transient_snes_mpfaof90

test-transient-th-mpfao :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/transient/transient_th_mpfao $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/transient/transient_th_mpfao.cfg \
		--logfile_prefix transient_th_mpfao

test-richards :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/richards/richards_driver $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/richards/richards.cfg \
		--logfile_prefix richards

test-mpfao :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/mpfao/mpfao $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/mpfao/mpfao.cfg \
		--logfile_prefix mpfao

test-steadyf90 :
	$(PYTHON) $(TEST_MANAGER) -e ../demo/steadyf90/steadyf90 $(TEST_OPTIONS) \
		--suite standard standard_parallel \
		--config-files ../demo/steadyf90/steadyf90.cfg \
		--logfile_prefix steadyf90

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

