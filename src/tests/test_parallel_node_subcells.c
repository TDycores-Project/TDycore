#include <tdycore_tests.h>
#include <tdycore.h>

// This unit test suite tests that each node in a periodic box mesh is connected
// to exactly 8 subcells in a parallel configuration.

static MPI_Comm comm;

// This function is called before the selected tests execute.
static void setup(int argc, char** argv) {
  PetscInitialize(&argc, &argv, NULL, NULL);
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
  // Create a 10x10x10 box mesh, periodic in all directions ("3-torus").
  PetscInt N = 10, ierr;
  DM mesh;
  PetscInt dim = 3;
  const PetscInt  faces[3] = {N,N,N};
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  const DMBoundaryType periodicity[3] = {DM_BOUNDARY_PERIODIC,
                                         DM_BOUNDARY_PERIODIC,
                                         DM_BOUNDARY_PERIODIC};

  // We create an "interpolated" DMPlex, which includes edges and faces in
  // addition to cells and vertices. These various mesh "points" are accessible
  // via "strata" at specific "heights":
  // Height | Type
  // 0      | cell
  // 1      | face
  // 2      | edge
  // 3      | vertex
  const PetscBool interpolate = PETSC_TRUE;
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, lower, upper,
                             periodicity, interpolate, &mesh);
  assert_int_equal(0, ierr);

  // Here we set the adjacency relations for the mesh so that nodes can see
  // their attached cells.
  ierr = DMSetBasicAdjacency(mesh, PETSC_TRUE, PETSC_TRUE);
  assert_int_equal(0, ierr);

  // Mark all boundary faces with a "boundary" label.
  DMCreateLabel(mesh, "boundary");
  DMGetLabel(mesh, "boundary", &label);
  DMPlexMarkBoundaryFaces(mesh, 1, label);
  DMPlexLabelComplete(mesh, label);

  // Distribute the mesh amongst the processes and construct ghost cells.
  {
    DM mesh_dist = NULL;
    ierr = DMPlexDistribute(mesh, 1, NULL, &mesh_dist);
    assert_int_equal(0, ierr);
    if (mesh_dist != NULL) {
      DMDestroy(&mesh);
      mesh = mesh_dist;
      DM mesh_with_ghosts = NULL;
      ierr = DMPlexConstructGhostCells(mesh, NULL, NULL, &mesh_with_ghosts);
      assert_int_equal(0, ierr);
      if (mesh_with_ghosts != NULL) {
        DMDestroy(&mesh);
        mesh = mesh_with_ghosts;
      }
    }
  }

  // Traverse the vertices of the mesh and check that they are connected to
  // the appropriate cells.
  PetscInt v_start, v_end;
  ierr = DMPlexGetHeightStratum(mesh, 3, &v_start, &v_end);
  assert_int_equal(0, ierr);
  for (PetscInt v = v_start; v < v_end; ++v) {
    // Get all points in the transitive closure for this vertex. The point array
    // is populated here each point and its orientation, so points[2*p] gives
    // the index of the the pth point.
    PetscInt num_points, *points = NULL;
    ierr = DMPlexGetTransitiveClosure(mesh, v, PETSC_FALSE, &num_points, &points);
    assert_int_equal(0, ierr);

    // Count up all the points that are cells (height == 0).
    PetscInt num_cells = 0;
    for (PetscInt p = 0; p < num_points; ++p) {
      PetscInt height;
      ierr = DMPlexGetPointHeight(mesh, points[2*p], &height);
      assert_int_equal(0, ierr);
      if (height == 0) {
        ++num_cells;
      }
    }

    // Put our toys away.
    ierr = DMPlexRestoreTransitiveClosure(mesh, v, PETSC_FALSE, &num_points, &points);
    assert_int_equal(0, ierr);

//    printf("vertex %d has %d subcells\n", v, num_cells);
    assert_int_equal(8, num_cells);
  }

  // Clean up.
  DMDestroy(&mesh);
}

int main(int argc, char* argv[])
{
  // Define our set of unit tests.
  const struct CMUnitTest tests[] =
  {
    cmocka_unit_test(TestPeriodicBoxMeshNodeSubcells),
  };

  // Run the selected tests on 3 processes.
  return run_selected_tests(argc, argv, setup, tests, breakdown, 3);
}
