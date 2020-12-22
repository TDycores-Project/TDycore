#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <string.h>
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

int main(int argc, char* argv[])
{
  // Stash command line arguments.
  argc_ = argc;
  argv_ = argv;

  const struct CMUnitTest tests[] =
  {
    cmocka_unit_test(TestTDyInit),
    cmocka_unit_test(TestPetscCommWorld),
    cmocka_unit_test(TestMPIAllreduce),
    cmocka_unit_test(TestTDyFinalize),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
