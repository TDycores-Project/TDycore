#include <tdycore_tests.h>
#include <tdycore.h>

// This unit test suite tests that each node in a periodic box mesh is connected
// to exactly 8 subcells in a parallel configuration.

static MPI_Comm comm;

// This function is called before the selected tests execute.
static void setup(int argc, char** argv) {
  PetscInitializeNoArguments();
  comm = PETSC_COMM_WORLD;
}

// This function is called at the end of the program.
static void breakdown() {
  PetscFinalize();
}

// This function creates a 3D domain-decomposed box mesh that's periodic in all
// directions and verifies that each of its nodes is attached to exactly 8
// subcells on each subdomain.
static void TestPeriodicBoxMeshNodeSubcells(void **state)
{
  // Create a 10x10x10 box mesh, periodic in all directions.
  PetscInt N = 10, ierr;
  DM dm;
  PetscInt dim = 3;
  const PetscInt  faces[3] = {N,N,N};
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  const DMBoundaryType periodicity[3] = {DM_BOUNDARY_PERIODIC,
                                         DM_BOUNDARY_PERIODIC,
                                         DM_BOUNDARY_PERIODIC};
  ierr = DMPlexCreateBoxMesh(comm,dim,PETSC_FALSE,faces,lower,upper,
                             periodicity,PETSC_TRUE,&dm);
  {
    DM dmDist = NULL;
    ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
    assert_int_equal(0, ierr);
    if (dmDist != NULL) {
      DMDestroy(&dm);
      dm = dmDist;
    }
  }

  // Clean up.
  DMDestroy(&dm);
}

int main(int argc, char* argv[])
{
  // Define our set of unit tests.
  const struct CMUnitTest tests[] =
  {
    cmocka_unit_test(TestPeriodicBoxMeshNodeSubcells),
  };

  // Run the selected tests on 3 processes.
  run_selected_tests(argc, argv, setup, tests, breakdown, 3);
}
