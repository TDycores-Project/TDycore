#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <string.h>
#include <stdlib.h>
#include <cmocka.h>

#include <tdycore.h>

// This unit test suite tests the initialization and finalization of the
// TDycore library.

// Globals for holding command line arguments.
static int argc_;
static char **argv_;

// Test whether TDyInit works and initializes MPI properly.
static void TestTDyInit(void **state)
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  assert_false(mpi_initialized);
  TDyInit(argc_, argv_);
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized);
}

// Test whether the PETSC_COMM_WORLD communicator behaves properly within cmocka.
static void TestPetscCommWorld(void **state)
{
  int num_procs;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);
  assert_true(num_procs >= 1);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  assert_true(rank >= 0);
  assert_true(rank < num_procs);
}

// Test whether MPI performs as expected within CMocka's environment.
static void TestMPIAllreduce(void **state)
{
  // Let's see if we can properly sum over all ranks.
  int num_procs;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);

  int one = 1;
  int sum;

  MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  assert_int_equal(num_procs, sum);
}

// Test whether TDyFinalize works as expected.
static void TestTDyFinalize(void **state)
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized);
  TDyFinalize();
}

static int _run_selected_tests(const char* command,
                               int num_tests,
                               const struct CMUnitTest tests[num_tests]) {
  // If we're asked for a count of the tests available, print that number to
  // stdout.
  if (command != NULL) {
    if (strcasecmp(command, "count") == 0) {
      fprintf(stdout, "%d\n", num_tests);
      exit(0);
    } else {
      // Try to interpret the argument as an index for the desired test.
      char *endptr;
      long index = strtol(command, &endptr, 10);
      if (*endptr == '\0') { // got a valid index!
        if ((index < 0) || (index >= num_tests)) {
          fprintf(stderr, "Invalid test index: %ld (must be in [0, %d])\n",
            index, num_tests);
          exit(1);
        } else {
          return cmocka_run_group_tests(&tests[index], NULL, NULL);
        }
      } else {
        fprintf(stderr, "Invalid argument: %s (must be 'count' or index)\n",
          argv[1]);
        exit(1);
      }
    }
  } else {
    // Just run all the tests in one go.
    return cmocka_run_group_tests(tests, NULL, NULL);
  }
}

#define run_selected_tests(argc, argv, tests) { \
  const char* command = (argc > 1) ? argv[1] : NULL; \
  int num_tests = sizeof(tests) / sizeof((tests)[0])); \
  _run_selected_tests(command, num_tests, tests)

int main(int argc, char* argv[])
{
  // Stash command line arguments.
  argc_ = argc;
  argv_ = argv;

  // Define our set of unit tests.
  const struct CMUnitTest tests[] =
  {
    cmocka_unit_test(TestTDyInit),
    cmocka_unit_test(TestPetscCommWorld),
    cmocka_unit_test(TestMPIAllreduce),
    cmocka_unit_test(TestTDyFinalize),
  };

  return run_selected_tests(argc, argv, tests);
}
