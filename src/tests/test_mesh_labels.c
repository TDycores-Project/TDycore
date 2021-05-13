#include <tdycore_tests.h>
#include <tdycore.h>

// This unit test suite tests the ability to read mesh files and add labels for
// specifying boundary conditions.

// Test whether we can write a box mesh to an Exodus file.
static void TestExodusWrite(void **state)
{
  // Create a box mesh.
  PetscInt N = 10, ierr;
  DM dm, dmDist = NULL;
  PetscInt dim = 3;
  const PetscInt  faces[3] = {N,N,N};
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
                             NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist); CHKERRQ(ierr);
  if (dmDist) {
    DMDestroy(&dm);
    dm = dmDist;
  }

  // Write it to an Exodus file.
  PetscViewer v;
  ierr = PetscViewerExodusIIOpen(PETSC_COMM_WORLD, "boxmesh.exo",
                                 FILE_MODE_WRITE, &v); CHKERRQ(ierr);
  ierr = DMView(dm, v); CHKERRQ(ierr);
  PetscViewerDestroy(&v);

  DMDestroy(&dm);
}

// Test whether we can read in an Exodus mesh file.
static void TestExodusRead(void **state)
{
  PetscInt ierr;

  // Read a DM from an Exodus file.
  PetscViewer v;
  ierr = PetscViewerExodusIIOpen(PETSC_COMM_WORLD, "boxmesh.exo",
                                 FILE_MODE_READ, &v); CHKERRQ(ierr);
  DM dm;
  ierr = DMCreate(PETSC_COMM_WORLD, &dm); CHKERRQ(ierr);
  DMSetType(dm, DMPLEX);
  DMLoad(dm, v);
  PetscViewerDestroy(&v);

  DMDestroy(&dm);
}

// Test whether we can successfully write out side sets for boundary conditions
// to an Exodus file.
static void TestExodusWriteSideSets(void **state)
{
}

// Test whether we can read an Exodus file we've written with side sets.
static void TestExodusReadSideSets(void **state)
{
}

int main(int argc, char* argv[])
{
  PetscInitializeNoArguments();

  // Define our set of unit tests.
  const struct CMUnitTest tests[] =
  {
    cmocka_unit_test(TestExodusWrite),
    cmocka_unit_test(TestExodusRead),
    cmocka_unit_test(TestExodusWriteSideSets),
    cmocka_unit_test(TestExodusReadSideSets),
  };

  run_selected_tests(argc, argv, tests, 1);
  PetscFinalize();
}
