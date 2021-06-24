#include <tdycore_tests.h>
#include <tdycore.h>

// We store a set of indices of locally-owned mesh points in an INT_SET.
#include <petsc/private/kernels/khash.h>
KHASH_SET_INIT_INT(INT_SET)

#ifdef CHKERRQ
#undef CHKERRQ
#endif
#define CHKERRQ(x) assert_int_equal(0, x)

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
  //
  // Height | Type
  // 0      | cell
  // 1      | face
  // 2      | edge
  // 3      | vertex
  //
  // (An "uninterpolated" mesh is just a cell-vertex mesh without edges and
  //  faces.)
  const PetscBool interpolate = PETSC_TRUE;
  PetscInt vertex_height = interpolate ? 3 : 1;
  PetscInt cell_height = 0;
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, lower, upper,
                             periodicity, interpolate, &mesh); CHKERRQ(ierr);

  // Here we set the adjacency relations for the mesh so that nodes can see
  // their attached cells ("FVM++").
  ierr = DMSetBasicAdjacency(mesh, PETSC_TRUE, PETSC_TRUE); CHKERRQ(ierr);

  // Distribute the mesh amongst all processes and gather the "local" vertices
  // that appear only on this process.
  khash_t(INT_SET)* local_vertices = kh_init(INT_SET);
  {
    DM mesh_dist = NULL;
    ierr = DMPlexDistribute(mesh, 0, NULL, &mesh_dist); CHKERRQ(ierr);
    if (mesh_dist != NULL) {
      DMDestroy(&mesh);
      mesh = mesh_dist;
    }

    PetscInt v_start, v_end;
    ierr = DMPlexGetHeightStratum(mesh, vertex_height, &v_start, &v_end); CHKERRQ(ierr);
    int retval;
    for (PetscInt v = v_start; v < v_end; ++v) {
      kh_put(INT_SET, local_vertices, v, &retval);
    }
  }

  // Add a layer of overlapping cells from other processes.
  {
    DM mesh_dist = NULL;
    ierr = DMPlexDistributeOverlap(mesh, 1, NULL, &mesh_dist); CHKERRQ(ierr);
    if (mesh_dist != NULL) {
      DMDestroy(&mesh);
      mesh = mesh_dist;
    }
  }

  // Traverse the **locally owned** vertices of the mesh and check that they are
  // connected to the appropriate cells.
  PetscInt v_start, v_end;
  ierr = DMPlexGetHeightStratum(mesh, vertex_height, &v_start, &v_end); CHKERRQ(ierr);
  for (PetscInt v = v_start; v < v_end; ++v) {

    PetscBool v_is_local = kh_exist(local_vertices, v);
    if (v_is_local) {
      // Get all points in the transitive closure for this vertex.
      PetscInt num_points, *points = NULL;
      ierr = DMPlexGetTransitiveClosure(mesh, v, PETSC_FALSE, &num_points, &points); CHKERRQ(ierr);

      // Count all the points that are cells. The points array is populated with
      // each point and its orientation, so points[2*p] gives the index of the
      // pth point.
      PetscInt num_cells = 0;
      for (PetscInt p = 0; p < num_points; ++p) {
        PetscInt height;
        PetscInt point = points[2*p];
        ierr = DMPlexGetPointHeight(mesh, point, &height); CHKERRQ(ierr);
        if (height == cell_height) {
          ++num_cells;
        }
      }

      // Put our toys away.
      ierr = DMPlexRestoreTransitiveClosure(mesh, v, PETSC_FALSE, &num_points, &points); CHKERRQ(ierr);

      // Every local vertex should see exactly 8 cells.
      assert_int_equal(8, num_cells);
    }
  }

  // Clean up.
  kh_destroy(INT_SET, local_vertices);
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
