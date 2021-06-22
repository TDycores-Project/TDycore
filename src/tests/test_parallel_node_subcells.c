#include <tdycore_tests.h>
#include <tdycore.h>

// We store a set of indices of locally-owned mesh points in an INT_SET.
#include <petsc/private/kernels/khash.h>
KHASH_SET_INIT_INT(INT_SET)

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
  // Note: An "uninterpolated" mesh is just a cell-vertex mesh without edges and
  // Note: faces.
  const PetscBool interpolate = PETSC_FALSE;
  PetscInt vertex_height = interpolate ? 3 : 1;
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, lower, upper,
                             periodicity, interpolate, &mesh);
  assert_int_equal(0, ierr);

  // Here we set the adjacency relations for the mesh so that nodes can see
  // their attached cells.
  ierr = DMSetBasicAdjacency(mesh, PETSC_TRUE, PETSC_TRUE);
  assert_int_equal(0, ierr);

  // Distribute the mesh amongst the processes and construct a single layer of
  // cells overlapping with other processes. Create a distributed star forest so
  // we can tell which vertices are owned by the local process ("roots").
  PetscSF sf;
  {
    DM mesh_dist = NULL;
    PetscSF sf_dist = NULL;
    ierr = DMPlexDistribute(mesh, 1, &sf, &mesh_dist);
    assert_int_equal(0, ierr);
    if (mesh_dist != NULL) {
      DMDestroy(&mesh);
      mesh = mesh_dist;

      ierr = DMPlexCreatePointSF(mesh, sf, PETSC_TRUE, &sf_dist);
      assert_int_equal(0, ierr);
      if (sf_dist != NULL) {
        PetscSFDestroy(&sf);
        sf = sf_dist;
      }
    }
    PetscSFSetUp(sf);
  }

  int my_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);

  // Get the indices of all mesh points owned by other processes and store them
  // in a set.
  khash_t(INT_SET)* remote_points = kh_init(INT_SET);
  {
    PetscInt nranks, num_leaves;
    const PetscMPIInt *ranks;
    const PetscInt *rank_offsets, *local_indices;
    ierr = PetscSFGetRootRanks(sf, &nranks, &ranks, &rank_offsets,
                               &local_indices, NULL);
    if (nranks == 0) {
    } else {
      for (int r = 0; r < nranks; ++r) {
        if (ranks[r] != my_rank) {
          PetscInt begin = rank_offsets[r];
          PetscInt end = rank_offsets[r+1];
          int retval;
          for (PetscInt i = begin; i < end; ++i) {
            kh_put(INT_SET, remote_points, local_indices[i], &retval);
          }
        }
      }
    }
  }

  // Traverse the **locally owned** vertices of the mesh and check that they are
  // connected to the appropriate cells.
  PetscInt v_start, v_end;
  ierr = DMPlexGetHeightStratum(mesh, vertex_height, &v_start, &v_end);
  assert_int_equal(0, ierr);
  for (PetscInt v = v_start; v < v_end; ++v) {

    // A vertex is owned locally if it's not in the set of remote points.
    PetscBool v_is_local = ((kh_size(remote_points) == 0) || !kh_exist(remote_points, v));
    if (v_is_local) {
      // Get all points in the transitive closure for this vertex. The point array
      // is populated here each point and its orientation, so points[2*p] gives
      // the index of the the pth point.
      PetscInt num_points, *points = NULL;
      ierr = DMPlexGetTransitiveClosure(mesh, v, PETSC_FALSE, &num_points, &points);
      assert_int_equal(0, ierr);

      // Count all the points that are cells (height == 0).
      PetscInt num_cells = 0;
      for (PetscInt p = 0; p < num_points; ++p) {
        PetscInt height;
        PetscInt point = points[2*p];
        //      assert_true(point < num_local_points);
        ierr = DMPlexGetPointHeight(mesh, point, &height);
        assert_int_equal(0, ierr);
        if (height == 0) {
          ++num_cells;
        }
      }

      // Put our toys away.
      ierr = DMPlexRestoreTransitiveClosure(mesh, v, PETSC_FALSE, &num_points, &points);
      assert_int_equal(0, ierr);

      assert_int_equal(8, num_cells);
    }
  }

  // Clean up.
  kh_destroy(INT_SET, remote_points);
  PetscSFDestroy(&sf);
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
