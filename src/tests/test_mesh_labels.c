#include <tdycore_tests.h>
#include <tdycore.h>
#include <petscviewerhdf5.h>
//#include <petscviewerexodusii.h>

// This unit test suite tests the ability to read mesh files and add labels for
// specifying boundary conditions.

static MPI_Comm comm;

static void setup(int argc, char** argv) {
  PetscInitializeNoArguments();
  comm = PETSC_COMM_WORLD;
}

static void breakdown() {
  PetscFinalize();
}

// Test whether we can write a box mesh to a mesh file.
static void TestMeshWrite(void **state)
{
  // Create a box mesh.
  PetscInt N = 10, ierr;
  DM dm;
  PetscInt dim = 3;
  const PetscInt  faces[3] = {N,N,N};
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  ierr = DMPlexCreateBoxMesh(comm,dim,PETSC_FALSE,faces,lower,upper,
                             NULL,PETSC_TRUE,&dm);
  assert_int_equal(0, ierr);

  // Write it to a mesh file.
  PetscViewer v;
  ierr = PetscViewerHDF5Open(comm, "boxmesh.hdf5", FILE_MODE_WRITE, &v);
  assert_int_equal(0, ierr);
  ierr = DMView(dm, v);
  assert_int_equal(0, ierr);
  PetscViewerDestroy(&v);

  DMDestroy(&dm);
}

// Test whether we can read in a mesh file.
static void TestMeshRead(void **state)
{
  PetscInt ierr;

  // Read a DM from a mesh file.
  PetscViewer v;
  ierr = PetscViewerHDF5Open(comm, "boxmesh.hdf5", FILE_MODE_READ, &v);
//  ierr = PetscViewerExodusIIOpen(comm, "boxmesh.exo", FILE_MODE_READ, &v);
  assert_int_equal(0, ierr);
  DM dm;
  ierr = DMCreate(comm, &dm);
  assert_int_equal(0, ierr);
  DMSetType(dm, DMPLEX);
  DMLoad(dm, v);
  PetscViewerDestroy(&v);

  // If we're here, things are good.
  DMDestroy(&dm);
}

// Test whether we can successfully write out side sets for boundary conditions
// to a mesh file.
static void TestMeshWriteLabel(void **state)
{
  PetscInt ierr;

  // Read our mesh.
  PetscViewer v;
  ierr = PetscViewerHDF5Open(comm, "boxmesh.hdf5", FILE_MODE_READ, &v);
//  ierr = PetscViewerExodusIIOpen(comm, "boxmesh.exo", FILE_MODE_READ, &v);
  assert_int_equal(0, ierr);
  DM dm;
  ierr = DMCreate(comm, &dm);
  assert_int_equal(0, ierr);
  DMSetType(dm, DMPLEX);
  DMLoad(dm, v);
  PetscViewerDestroy(&v);

  // Now mark boundary faces with a label.
  DMLabel label;
  DMCreateLabel(dm, "boundary");
  DMGetLabel(dm, "boundary", &label);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label);
  assert_int_equal(0, ierr);
  ierr = DMPlexLabelComplete(dm, label);
  assert_int_equal(0, ierr);

  // Write out the mesh with boundary faces labeled.
  ierr = PetscViewerHDF5Open(comm, "boxmesh_labeled.hdf5", FILE_MODE_WRITE, &v);
//  ierr = PetscViewerExodusIIOpen(comm, "boxmesh_labeled.exo", FILE_MODE_WRITE, &v);
  assert_int_equal(0, ierr);
  ierr = DMView(dm, v);
  assert_int_equal(0, ierr);
  PetscViewerDestroy(&v);

  DMDestroy(&dm);
}

// Test whether we can read a mesh file we've written with side sets.
static void TestMeshReadLabel(void **state)
{
  PetscInt ierr;

  // Read the boundary-labeled mesh.
  PetscViewer v;
  ierr = PetscViewerHDF5Open(comm, "boxmesh_labeled.hdf5", FILE_MODE_READ, &v);
//  ierr = PetscViewerExodusIIOpen(comm, "boxmesh_labeled.exo", FILE_MODE_READ, &v);
  assert_int_equal(0, ierr);
  DM dm;
  ierr = DMCreate(comm, &dm);
  assert_int_equal(0, ierr);
  DMSetType(dm, DMPLEX);
  DMLoad(dm, v);
  PetscViewerDestroy(&v);

  // Traverse the boundary faces and compare them to a newly-generated boundary.
  DMLabel label, new_label;
  DMGetLabel(dm, "boundary", &label);
  DMCreateLabel(dm, "new_boundary");
  DMGetLabel(dm, "new_boundary", &new_label);
  DMPlexMarkBoundaryFaces(dm, 1, new_label);
  IS label_IS, new_label_IS;
  if (label) {
    ierr = DMLabelGetValueIS(label, &label_IS);
    ierr = DMLabelGetValueIS(new_label, &new_label_IS);
    PetscInt num_faces, num_new_faces;
    ierr = ISGetSize(label_IS, &num_faces);
    assert_int_equal(0, ierr);
    ierr = ISGetSize(label_IS, &num_new_faces);
    assert_int_equal(0, ierr);
    assert_int_equal(num_faces, num_new_faces);
    const PetscInt *faces, *new_faces;
    ierr = ISGetIndices(label_IS, &faces);
    assert_int_equal(0, ierr);
    ierr = ISGetIndices(new_label_IS, &new_faces);
    assert_int_equal(0, ierr);
    for (PetscInt f = 0; f < num_faces; ++f) {
      assert_int_equal(faces[f], new_faces[f]);
    }
    ISRestoreIndices(label_IS, &faces);
    ISRestoreIndices(new_label_IS, &new_faces);
    ISDestroy(&label_IS);
    ISDestroy(&new_label_IS);
  }

  DMDestroy(&dm);
}

int main(int argc, char* argv[])
{
  // Define our set of unit tests.
  const struct CMUnitTest tests[] =
  {
    cmocka_unit_test(TestMeshWrite),
    cmocka_unit_test(TestMeshRead),
    cmocka_unit_test(TestMeshWriteLabel),
    cmocka_unit_test(TestMeshReadLabel),
  };

  return run_selected_tests(argc, argv, setup, tests, breakdown, 1);
}
