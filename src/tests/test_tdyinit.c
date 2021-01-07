#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cmocka.h>

#include <tdycore.h>

// This unit test suite tests the initialization and finalization of the
// TDycore library.

// Globals for holding command line arguments.
static int argc_;
static char **argv_;

// Flag indicating whether tests are run in their own process.
static bool isolate_tests = false;

// Test whether TDyInit works and initializes MPI properly.
static void TestTDyInit(void **state)
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  assert_false(mpi_initialized);
  TDyInit(argc_, argv_);
  MPI_Initialized(&mpi_initialized);
  assert_true(mpi_initialized);

  if (isolate_tests) {
    TDyFinalize();
  }
}

// Test whether the PETSC_COMM_WORLD communicator behaves properly within cmocka.
static void TestPetscCommWorld(void **state)
{
  if (isolate_tests) {
    TDyInit(argc_, argv_);
  }

  int num_procs;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);
  assert_true(num_procs >= 1);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  assert_true(rank >= 0);
  assert_true(rank < num_procs);

  if (isolate_tests) {
    TDyFinalize();
  }
}

// Test whether MPI performs as expected within CMocka's environment.
static void TestMPIAllreduce(void **state)
{
  if (isolate_tests) {
    TDyInit(argc_, argv_);
  }

  // Let's see if we can properly sum over all ranks.
  int num_procs;
  MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);

  int one = 1;
  int sum;

  MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  assert_int_equal(num_procs, sum);

  // This is the last test, so we finalize no matter what.
  TDyFinalize();
}

static int _run_selected_tests(const char* command,
                               int num_tests,
                               const struct CMUnitTest tests[num_tests]) {
  // Set our test isolation flag.
  isolate_tests = true;

  if (strcasecmp(command, "count") == 0) { // asked for # of tests
    fprintf(stdout, "%d\n", num_tests);
    return 0;
  } else { // asked for a specific test index
    // Try to interpret the argument as an index for the desired test.
    char *endptr;
    long index = strtol(command, &endptr, 10);
    if (*endptr == '\0') { // got a valid index!
      if ((index < 0) || (index >= num_tests)) {
        fprintf(stderr, "Invalid test index: %ld (must be in [0, %d])\n",
            index, num_tests);
        return 1;
      } else {
        const struct CMUnitTest selected_tests[] = { tests[index] };
        return cmocka_run_group_tests(selected_tests, NULL, NULL);
      }
    } else {
      fprintf(stderr, "Invalid command: %s (must be 'count' or index)\n",
          command);
      return 1;
    }
  }
}

#define run_selected_tests(argc, argv, tests) \
  (argc > 1) ? _run_selected_tests(argv[1], \
                                   sizeof(tests) / sizeof((tests)[0]), \
                                   tests) \
             : cmocka_run_group_tests(tests, NULL, NULL)

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
    cmocka_unit_test(TestMPIAllreduce)
  };

  run_selected_tests(argc, argv, tests);
}
