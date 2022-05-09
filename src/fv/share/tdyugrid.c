#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <private/tdyugridimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>

static PetscInt vertex_separator = -777;
static PetscInt dual_separator = -888;
static PetscInt cell_separator = -999999;


PetscErrorCode TDyUGridCreate(TDyUGrid *ugrid) {

  ugrid = malloc(sizeof(TDyUGrid));

  ugrid->num_cells_global = 0;
  ugrid->num_cells_local = 0;
  ugrid->num_cells_ghost = 0;

  ugrid->max_verts_per_cell = 0;
  ugrid->max_ndual_per_cell = 0;

  ugrid->num_verts_global = 0;
  ugrid->num_verts_local = 0;
  ugrid->num_verts_natural = 0;

  ugrid->global_offset = 0;

  ugrid->cell_vertices = NULL;
  ugrid->cell_num_vertices = NULL;

  ugrid->cell_ids_natural = NULL;
  ugrid->cell_ids_petsc = NULL;
  ugrid->ghost_cell_ids_petsc = NULL;

  ugrid->cell_neighbors_ghosted = NULL;
  ugrid->cell_num_neighbors_ghosted = NULL;

  ugrid->vertices = NULL;
  ugrid->vertex_ids_natural = NULL;

  ugrid->ao_natural_to_petsc = NULL;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode ReadPFLOTRANMeshFile(const char *mesh_file, TDyUGrid *ugrid){

  PetscViewer viewer;
  Vec cells, verts;
  PetscScalar *v_p;
  PetscErrorCode ierr;

  PetscInt ***cell_vertices = &ugrid->cell_vertices;
  PetscReal ***vertices = &ugrid->vertices;
  PetscInt *num_cells_local = &ugrid->num_cells_local;
  PetscInt *max_verts_per_cell = &ugrid->max_verts_per_cell;
  PetscInt *num_verts_local = &ugrid->num_verts_local;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, mesh_file, FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/Domain");CHKERRQ(ierr);

  // Read cells
  ierr = VecCreate(PETSC_COMM_WORLD, &cells); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cells, "cells");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cells, "Cells"); CHKERRQ(ierr);
  ierr = VecSetFromOptions(cells); CHKERRQ(ierr);
  ierr = VecLoad(cells,viewer); CHKERRQ(ierr);

  // Read vertices
  ierr = VecCreate(PETSC_COMM_WORLD, &verts); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) verts, "vertices");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) verts, "Vertices"); CHKERRQ(ierr);
  ierr = VecSetFromOptions(verts); CHKERRQ(ierr);
  ierr = VecLoad(verts,viewer); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // Save information about cells
  PetscInt vec_cells_size;
  ierr = VecGetSize(cells, &ugrid->num_cells_global);
  ierr = VecGetLocalSize(cells, &vec_cells_size);
  PetscInt blocksize;
  ierr = VecGetBlockSize(cells, &blocksize);
  *num_cells_local = (PetscInt) (vec_cells_size/blocksize);

  *max_verts_per_cell = blocksize - 1;
  ierr = TDyAllocate_IntegerArray_2D(cell_vertices, *num_cells_local, *max_verts_per_cell); CHKERRQ(ierr);

  ierr = VecGetArray(cells, &v_p); CHKERRQ(ierr);
  PetscInt count=0;

  MPI_Comm comm;
  ierr = PetscObjectGetComm((PetscObject)cells, &comm); CHKERRQ(ierr);
  for (PetscInt i=0; i<*num_cells_local; i++) {
    PetscInt nvertex = (PetscInt) v_p[count++];
    switch (nvertex) {
      case 4:
        break;
      case 5:
        break;
      case 6:
        break;
      case 8:
        break;
      default:
        SETERRQ(comm,PETSC_ERR_USER,"Unknown cell type");
    }
    for (PetscInt j=1; j<*max_verts_per_cell+1; j++) {
      (*cell_vertices)[i][j-1] = (PetscInt) v_p[count++] - 1; // Converting PFLOTRAN's 1-based index to 0-based index
    }
  }
  ierr = VecRestoreArray(cells, &v_p); CHKERRQ(ierr);
  ierr = VecDestroy(&cells); CHKERRQ(ierr);

  // Save information about vertices
  PetscInt vert_dim, vec_verts_size;
  ierr = VecGetSize(verts, &ugrid->num_verts_global);
  ierr = VecGetLocalSize(verts, &vec_verts_size);
  ierr = VecGetBlockSize(verts, &vert_dim);
  if (vert_dim != 3) {
    ierr = PetscObjectGetComm((PetscObject)verts, &comm); CHKERRQ(ierr);
    SETERRQ(comm,PETSC_ERR_USER,"Vertices in the the HDF5 is not 3D");
  }
  *num_verts_local = (PetscInt) (vec_verts_size/vert_dim);

  ierr = TDyAllocate_RealArray_2D(vertices, *num_verts_local, vert_dim); CHKERRQ(ierr);
  ierr = VecGetArray(verts, &v_p); CHKERRQ(ierr);
  count=0;
  for (PetscInt i=0; i<*num_verts_local; i++) {
    for (PetscInt j=0; j<vert_dim; j++) {
      (*vertices)[i][j] = v_p[count++];
    }
  }
  ierr = VecRestoreArray(verts, &v_p); CHKERRQ(ierr);

  ierr = VecDestroy(&verts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode UGridPrintCells(TDyUGrid *ugrid) {

  PetscInt rank, commsize;

  MPI_Comm_size(PETSC_COMM_WORLD, &commsize);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt **cell_vertices = ugrid->cell_vertices;
  if (rank == 0) printf("Cells:\n");
  for (PetscInt irank=0; irank<commsize; irank++) {
    if (rank == irank) {
      printf("Rank = %d\n",rank);
      for (PetscInt icell=0; icell<ugrid->num_cells_local; icell++) {
        for (PetscInt ivertex=0; ivertex<ugrid->max_verts_per_cell; ivertex++) {
          printf("%02d ",cell_vertices[icell][ivertex]);
        }
        printf("\n");
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode UGridPrintVertices(TDyUGrid *ugrid) {

  PetscInt rank, commsize;

  MPI_Comm_size(PETSC_COMM_WORLD, &commsize);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscReal **vertices = ugrid->vertices;
  if (rank == 0) printf("Vertices:\n");
  for (PetscInt irank=0; irank<commsize; irank++) {
    if (rank == irank) {
      printf("Rank = %d\n",rank);
      PetscInt dim=3;
      for (PetscInt ivertex=0; ivertex<ugrid->num_verts_local; ivertex++) {
        for (PetscInt idim=0; idim<dim; idim++) {
          printf("%e ",vertices[ivertex][idim]);
        }
        printf("\n");
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode DetermineMaxNumVerticesActivePerCell(TDyUGrid *ugrid) {

  PetscErrorCode ierr;

  PetscInt nvert_active = 0;
  for (PetscInt icell=0; icell<ugrid->num_cells_local; icell++) {
    PetscInt tmp=0;
    for (PetscInt ivertex=0; ivertex<ugrid->max_verts_per_cell; ivertex++) {
      if (ugrid->cell_vertices[icell][ivertex] > 0) {
        tmp++;
      }
    }
    if (tmp > nvert_active){
      nvert_active = tmp;
    }
  }

  ierr = MPI_Allreduce(&nvert_active, &ugrid->max_verts_per_cell, 1, MPI_INTEGER, MPI_MAX, PETSC_COMM_WORLD); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode CreateAdjacencyMatrix(TDyUGrid *ugrid, Mat *AdjMat) {

  PetscErrorCode ierr;

  // Determine the i-th and j-th values for creating the adjacency matrix:
  //   - 	i: the indices into j for the start of each row
  //   - 	j: the column indices for each row (sorted for each row).
  PetscInt *i, *j;
  PetscInt nrow = ugrid->num_cells_local;
  PetscInt ncol = ugrid->max_verts_per_cell;

  ierr = TDyAllocate_IntegerArray_1D(&i, nrow + 1); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&j, nrow*ncol); CHKERRQ(ierr);

  i[0] = 0;
  PetscInt count=0;
  for (PetscInt icell=0; icell<nrow; icell++) {
    for (PetscInt ivert=0; ivert<ncol; ivert++) {
      if (ugrid->cell_vertices[icell][ivert] >= 0 ){
      j[count++] = ugrid->cell_vertices[icell][ivert];
      }
    }
    i[icell+1] = count;
  }

  // Determine the global offset for rows on each rank
  PetscInt global_offset = 0;
  ierr = MPI_Exscan(&nrow, &global_offset, 1, MPI_INTEGER, MPI_SUM, PETSC_COMM_WORLD); CHKERRQ(ierr);

  ierr = MatCreateMPIAdj(PETSC_COMM_WORLD, nrow, ugrid->num_verts_global, i, j, PETSC_NULL, AdjMat); CHKERRQ(ierr);
  ierr = TDySavePetscMatAsASCII(*AdjMat, "Adj.out"); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode PartitionGrid(Mat DualMat, IS *NewCellRankIS, PetscInt *NewNumCellsLocal) {

   PetscErrorCode ierr;

  // Create partitioning object based on the dual matrix
  MatPartitioning Part;
  ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &Part); CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(Part, DualMat);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(Part);CHKERRQ(ierr);

  // Now partition the mesh
  ierr = MatPartitioningApply(Part, NewCellRankIS); CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&Part); CHKERRQ(ierr);

  // Compute the number of local grid cells on each processor
  PetscInt commsize, myrank;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &commsize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &myrank); CHKERRQ(ierr);

  PetscInt cell_counts[commsize];
  ierr = ISPartitioningCount(*NewCellRankIS, commsize, cell_counts); CHKERRQ(ierr);
  *NewNumCellsLocal = cell_counts[myrank];

  ierr = TDySavePetscISAsASCII(*NewCellRankIS,"is.out");

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode DetermineMaxNumDualCells(TDyUGrid *ugrid, Mat DualMat) {

  PetscErrorCode ierr;

  const PetscInt *ia_ptr, *ja_ptr;
  PetscBool success;
  PetscInt num_rows;
  ierr = MatGetRowIJ(DualMat, 0, PETSC_FALSE, PETSC_FALSE, &num_rows, &ia_ptr, &ja_ptr, &success);CHKERRQ(ierr);

  if (!success) {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)DualMat, &comm); CHKERRQ(ierr);
    SETERRQ(comm, PETSC_ERR_USER, "Error get row and column indices from dual matrix");
  }

  ugrid->max_ndual_per_cell = 0;
  for (PetscInt icell=0; icell<ugrid->num_cells_local; icell++) {
    PetscInt istart = ia_ptr[icell];
    PetscInt iend = ia_ptr[icell+1];
    PetscInt num_cols = iend - istart;
    if (num_cols > ugrid->max_ndual_per_cell) {
      ugrid->max_ndual_per_cell = num_cols;
    }
  }

  PetscInt tmp = ugrid->max_ndual_per_cell;
  ierr = MPI_Allreduce(&tmp, &ugrid->max_ndual_per_cell, 1, MPI_INTEGER, MPI_MAX, PETSC_COMM_WORLD);

  ierr = MatRestoreRowIJ(DualMat, 0, PETSC_FALSE, PETSC_FALSE, &num_rows, &ia_ptr, &ja_ptr, &success);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
static PetscErrorCode CreateISNatOrderToPetscOrder(TDyUGrid *ugrid, IS NewCellRankIS, PetscInt stride, IS *NatToPetscIS) {

  PetscErrorCode ierr;

  IS NumberingIS;
  const PetscInt *is_ptr;

  ierr = ISPartitioningToNumbering(NewCellRankIS, &NumberingIS); CHKERRQ(ierr);
  ierr = ISGetIndices(NumberingIS, &is_ptr); CHKERRQ(ierr);

  PetscInt num_cells_local_old = ugrid->num_cells_local;
  ierr = ISCreateBlock(PETSC_COMM_WORLD, stride, num_cells_local_old, is_ptr, PETSC_COPY_VALUES, NatToPetscIS); CHKERRQ(ierr);

  ierr = ISRestoreIndices(NumberingIS, &is_ptr); CHKERRQ(ierr);
  ierr = ISDestroy(&NumberingIS); CHKERRQ(ierr);

  ierr = TDySavePetscISAsASCII(NewCellRankIS,"is_scatter_elem_old_to_new.out");

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode CreateVectorNatOrder(TDyUGrid *ugrid, IS NewCellRankIS, PetscInt stride, Vec *NatOrderVec) {

  PetscErrorCode ierr;

  PetscInt num_cells_local_old = ugrid->num_cells_local;
  ierr = VecCreate(PETSC_COMM_WORLD, NatOrderVec); CHKERRQ(ierr);
  ierr = VecSetSizes(*NatOrderVec, stride*num_cells_local_old, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*NatOrderVec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode PackNatOrderVector(TDyUGrid *ugrid, Mat DualMat, Vec *NatOrderVec) {

  PetscErrorCode ierr;

  const PetscInt *ia_ptr, *ja_ptr;
  PetscBool success;
  PetscInt num_rows;
  ierr = MatGetRowIJ(DualMat, 0, PETSC_FALSE, PETSC_FALSE, &num_rows, &ia_ptr, &ja_ptr, &success);CHKERRQ(ierr);

  PetscScalar *v_ptr;
  ierr = VecGetArray(*NatOrderVec, &v_ptr); CHKERRQ(ierr);

  PetscInt count=0;
  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt global_offset = 0;
  ierr = MPI_Exscan(&ugrid->num_cells_local, &global_offset, 1, MPI_INTEGER, MPI_SUM, PETSC_COMM_WORLD); CHKERRQ(ierr);
  ugrid->global_offset = global_offset;

  for (PetscInt icell=0; icell<ugrid->num_cells_local; icell++){
    v_ptr[count++] = -global_offset-icell-1;
    v_ptr[count++] = vertex_separator;

    for (PetscInt ivertex=0; ivertex<ugrid->max_verts_per_cell; ivertex++){
      v_ptr[count++] = ugrid->cell_vertices[icell][ivertex] + 1; // increment to 1-based ordering
    }
    v_ptr[count++] = dual_separator;

    PetscInt istart = ia_ptr[icell];
    PetscInt iend = ia_ptr[icell+1];
    PetscInt num_cols = iend - istart;

    for (PetscInt icol=0; icol<ugrid->max_ndual_per_cell; icol++) {
      if (icol < num_cols) {
        v_ptr[count++] = ja_ptr[istart + icol] + 1; // increment to 1-based ordering
      } else {
        v_ptr[count++] = 0;
      }
    }

    v_ptr[count++] = cell_separator;
  }

  ierr = VecRestoreArray(*NatOrderVec, &v_ptr); CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(DualMat, 0, PETSC_FALSE, PETSC_FALSE, &num_rows, &ia_ptr, &ja_ptr, &success);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode PrePartNatOrder_To_PostPartNatOrder(PetscInt stride, PetscInt NewNumCellsLocal, IS *NatToPetscIS, Vec *Pre, Vec *Post) {

  PetscErrorCode ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, Post); CHKERRQ(ierr);
  ierr = VecSetSizes(*Post, stride*NewNumCellsLocal, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*Post); CHKERRQ(ierr);

  VecScatter VecScatter;
  ierr = VecScatterCreate(*Pre, PETSC_NULL, *Post, *NatToPetscIS, &VecScatter); CHKERRQ(ierr);
  ierr = VecScatterBegin(VecScatter, *Pre, *Post, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(VecScatter, *Pre, *Post, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&VecScatter); CHKERRQ(ierr);

  ierr = TDySavePetscVecAsASCII(*Pre,"elements_old.out");
  ierr = TDySavePetscVecAsASCII(*Post,"elements_natural.out");

 PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode SaveNaturalCellIDs(TDyUGrid *ugrid, Vec PostPartNatOrderVec, PetscInt NewNumCellsLocal, PetscInt stride) {

  PetscErrorCode ierr;

  ierr = TDyAllocate_IntegerArray_1D(&ugrid->cell_ids_natural, NewNumCellsLocal); CHKERRQ(ierr);

  PetscScalar *v_ptr;
  ierr = VecGetArray(PostPartNatOrderVec, &v_ptr); CHKERRQ(ierr);
  for (PetscInt icell=0; icell<NewNumCellsLocal; icell++) {
    ugrid->cell_ids_natural[icell] = -(PetscInt) v_ptr[icell*stride] - 1;
  }
  ierr = VecRestoreArray(PostPartNatOrderVec, &v_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode CreateApplicationOrder(TDyUGrid *ugrid, PetscInt NewGlobalOffset, PetscInt NewNumCellsLocal) {

  PetscErrorCode ierr;

  PetscInt int_array[NewNumCellsLocal];
  for (PetscInt icell=0; icell<NewNumCellsLocal; icell++) {
    int_array[icell] = icell + NewGlobalOffset;
  }

  ierr = AOCreateBasic(PETSC_COMM_WORLD, NewNumCellsLocal, ugrid->cell_ids_natural, int_array, &ugrid->ao_natural_to_petsc); CHKERRQ(ierr);
  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "ao.out", &viewer); CHKERRQ(ierr);
  ierr = AOView(ugrid->ao_natural_to_petsc, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode CellAndDualIDs_FromNatOrder_To_PetscOrder(TDyUGrid *ugrid, PetscInt stride, PetscInt dual_offset, PetscInt NewNumCellsLocal, Vec *PetscOrderVec) {

  PetscErrorCode ierr;

  PetscInt max_ndual = ugrid->max_ndual_per_cell;
  PetscInt size = NewNumCellsLocal * max_ndual;
  PetscInt IDs[size];
  PetscInt ndual = 0;

  PetscScalar *v_ptr;

  ierr = VecGetArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
  for (PetscInt icell=0; icell<NewNumCellsLocal; icell++) {

    IDs[ndual++] = ugrid->cell_ids_natural[icell]; // Are in 0-based index

    for (PetscInt idual=0; idual<max_ndual; idual++) {
      PetscInt dualID = (PetscInt) v_ptr[icell*stride + idual + dual_offset];
      if (dualID>0) {
        IDs[ndual++] = dualID-1; // Changing from 1-based index to 0-based index
      }
    }
  }
  ierr = VecRestoreArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);

  ierr = AOApplicationToPetsc(ugrid->ao_natural_to_petsc, ndual, IDs); CHKERRQ(ierr);

  ierr = VecGetArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&ugrid->cell_ids_petsc, NewNumCellsLocal); CHKERRQ(ierr);
  ndual = 0;

  for (PetscInt icell=0; icell<NewNumCellsLocal; icell++) {

    ugrid->cell_ids_petsc[icell] = IDs[ndual];
    v_ptr[icell*stride] =  IDs[ndual] + 1; // Changing from 0-based to 1-based

    for (PetscInt idual=0; idual<max_ndual; idual++) {
      PetscInt dualID = (PetscInt) v_ptr[icell*stride + idual + dual_offset];
      if (dualID > 0) {
        ndual++;
        v_ptr[icell*stride + idual + dual_offset] = IDs[ndual] + 1; // Changing from 0-based to 1-based
      }
    }
    ndual++;
  }
  ierr = VecRestoreArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsASCII(*PetscOrderVec,"elements_petsc.out");


  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode DualIDs_FromPetscOrder_To_LocalOrder(TDyUGrid *ugrid, PetscInt stride, PetscInt dual_offset, PetscInt NewNumCellsLocal, PetscInt NewGlobalOffset, Vec *PetscOrderVec) {

  PetscErrorCode ierr;

  PetscInt max_ndual = ugrid->max_ndual_per_cell;

  PetscScalar *v_ptr;

  ierr = VecGetArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);

  // Determine the maximum number of ghost cells
  PetscInt NumCellsGhost = 0;
  for (PetscInt icell=0; icell<NewNumCellsLocal; icell++){
    for (PetscInt idual=0; idual<max_ndual; idual++){
      PetscInt dualID = (PetscInt) v_ptr[icell*stride + idual + dual_offset];
      if (dualID > 0) {
        if (dualID <= NewGlobalOffset || dualID > NewGlobalOffset + NewNumCellsLocal) {
          NumCellsGhost++;
        }
      }
    }
  }

  PetscInt IntArray1[NumCellsGhost];

  // 1. Make a list of ghost cells IDs that in PETSc-order
  // 2. Change IDs of duals that are locally-owned and ghost cells in PetscOrderVec
  //   - Locally-owned dual IDs are positive values
  //   - Ghost dual IDs are negative values

  NumCellsGhost = 0;
  for (PetscInt icell=0; icell<NewNumCellsLocal; icell++){
    for (PetscInt idual=0; idual<max_ndual; idual++){
      PetscInt dualID = (PetscInt) v_ptr[icell*stride + idual + dual_offset];

      if (dualID > 0) {
        // Determine if the dual is a ghost based on the ID of the dual that is in PETSc-order
        if (dualID <= NewGlobalOffset || dualID > NewGlobalOffset + NewNumCellsLocal) {
          IntArray1[NumCellsGhost++] = dualID; // Save the dual ID in PETSc-order
          v_ptr[icell*stride + idual + dual_offset] = -NumCellsGhost;
        } else {
          v_ptr[icell*stride + idual + dual_offset] = dualID - NewGlobalOffset;
        }
      }
    }
  }

  ierr = VecRestoreArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
  ierr = TDySavePetscVecAsASCII(*PetscOrderVec,"elements_local_dual_unsorted.out");

  // Sort any ghost cells that are present
  if (NumCellsGhost >0) {

    // Example of sorting of ghost dual cell IDs:
    //
    // Index  Dual_ID  IntArray1  Sorted_IntArray1  IntArray2  IntArray3  IntArray4  IntArray5
    //   0      -1        11            09              1         09         0          2
    //   1      -2        09            10              2         10         1          0
    //   2      -3        10            11              0         11         2          1
    //   3      -4        12            11              5         12         2          4
    //   4      -5        13            12              3         13         3          5
    //   5      -1        11            13              4         -          4          3
    //
    //
    //  IntArray1  ---(IntArray2)---> Sorted_IntArray1  <--- IntArray4 --- IntArray3
    //  IntArray1 <---(IntArray5)---  Sorted_IntArray1
    //

    PetscInt IntArray2[NumCellsGhost];
    PetscInt IntArray3[NumCellsGhost];
    PetscInt IntArray4[NumCellsGhost];
    PetscInt IntArray5[NumCellsGhost];

    for (PetscInt ighost=0; ighost<NumCellsGhost; ighost++) {
      IntArray1[ighost] = IntArray1[ighost] - 1; // Converting to 0-based index
      IntArray2[ighost] = ighost;
      IntArray3[ighost] = -1;
      IntArray4[ighost] = -1;
      IntArray5[ighost] = -1;
    }

    ierr = PetscSortIntWithPermutation(NumCellsGhost, IntArray1, IntArray2); CHKERRQ(ierr);

    PetscInt tmp_int=0;

    PetscInt idx=IntArray2[tmp_int];
    IntArray3[tmp_int] = IntArray1[idx];

    for (PetscInt ighost=0; ighost<NumCellsGhost; ighost++) {

      PetscInt idx=IntArray2[ighost];
      if (IntArray3[tmp_int] < IntArray1[idx]) {
        tmp_int++;
        IntArray3[tmp_int] = IntArray1[idx];
      }

      IntArray4[ighost] = tmp_int;
      IntArray5[IntArray2[ighost]] = ighost;
    }

    NumCellsGhost = tmp_int+1;
    ierr = TDyAllocate_IntegerArray_1D(&ugrid->ghost_cell_ids_petsc, NumCellsGhost); CHKERRQ(ierr);
    for (PetscInt ighost=0; ighost<NumCellsGhost; ighost++) {
      ugrid->ghost_cell_ids_petsc[ighost] = IntArray3[ighost];
    }

    ierr = VecGetArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
    for (PetscInt icell=0; icell<NewNumCellsLocal; icell++){
      for (PetscInt idual=0; idual<max_ndual; idual++){
        PetscInt dualID = (PetscInt) v_ptr[icell*stride + idual + dual_offset];

        if (dualID < 0) {
          PetscInt idx = IntArray5[-dualID-1];
          v_ptr[icell*stride + idual + dual_offset] =  IntArray4[idx] + NewNumCellsLocal + 1; // converting to 1-based
        }
      }
    }
    ierr = VecRestoreArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
  }
  ierr = TDySavePetscVecAsASCII(*PetscOrderVec,"elements_local_dual.out");

  ugrid->num_cells_local = NewNumCellsLocal;
  ugrid->num_cells_ghost = NumCellsGhost;
  ugrid->num_cells_global = NewNumCellsLocal + NumCellsGhost;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode UpdateNaturalCellIDs(TDyUGrid *ugrid, PetscInt stride, PetscInt dual_offset, Vec *NatOrderVec, Vec *PetscOrderVec) {

  PetscErrorCode ierr;

  PetscInt nlmax = ugrid->num_cells_local;
  PetscInt ngmax = ugrid->num_cells_global;

  PetscInt cell_ids_natural_tmp[nlmax];

  for (PetscInt ilocal=0; ilocal<nlmax; ilocal++) {
    cell_ids_natural_tmp[ilocal] = ugrid->cell_ids_natural[ilocal];
  }

  PetscInt rank; MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  free(ugrid->cell_ids_natural);
  ierr = TDyAllocate_IntegerArray_1D(&ugrid->cell_ids_natural, ngmax); CHKERRQ(ierr);

  for (PetscInt ilocal=0; ilocal<nlmax; ilocal++) {
    ugrid->cell_ids_natural[ilocal] = cell_ids_natural_tmp[ilocal];
  }

  PetscInt max_ndual = ugrid->max_ndual_per_cell;
  PetscScalar *v_ptr, *v_ptr2;
  ierr = VecGetArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(*NatOrderVec, &v_ptr2); CHKERRQ(ierr);
  for (PetscInt icell=0; icell<nlmax; icell++){
    for (PetscInt idual=0; idual<max_ndual; idual++){
      PetscInt dualID = (PetscInt) v_ptr[icell*stride + idual + dual_offset]; // 1-based index

      if (dualID >= 0) {
        if (dualID > nlmax) {
          PetscInt natID = (PetscInt) v_ptr2[icell*stride + idual + dual_offset];
          ugrid->cell_ids_natural[dualID-1] = natID - 1; // Convert to 0-based index
        }
      }
    }
  }
  ierr = VecRestoreArray(*NatOrderVec, &v_ptr2); CHKERRQ(ierr);
  ierr = VecRestoreArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode DetermineNeigbhorsCellIDsInGhostedOrder(TDyUGrid *ugrid, PetscInt stride, PetscInt dual_offset, Vec *PetscOrderVec) {

  PetscErrorCode ierr;

  PetscInt max_ndual = ugrid->max_ndual_per_cell;
  PetscInt nlmax = ugrid->num_cells_local;

  ierr = TDyAllocate_IntegerArray_2D(&ugrid->cell_neighbors_ghosted, nlmax, max_ndual); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_1D(&ugrid->cell_num_neighbors_ghosted, nlmax); CHKERRQ(ierr);

  PetscScalar *v_ptr;
  ierr = VecGetArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);
  for (PetscInt icell=0; icell<nlmax; icell++){

    PetscInt count = 0;
    ugrid->cell_num_neighbors_ghosted[icell] = count; // initialize the number of neighbors

    for (PetscInt idual=0; idual<max_ndual; idual++){
      PetscInt dualID = (PetscInt) v_ptr[icell*stride + idual + dual_offset]; // 1-based index

      if (dualID > 0) {
        count++;

        if (dualID > nlmax) {
          dualID = -dualID+1; // Converting to 0-based index
        } else {
          dualID--; // Converting to 0-based index
        }
        ugrid->cell_neighbors_ghosted[icell][idual] = dualID;
      }
      ugrid->cell_num_neighbors_ghosted[icell] = count;
    }
  }
  ierr = VecRestoreArray(*PetscOrderVec, &v_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ScatterVecNatOrderToPetscOrder(TDyUGrid *ugrid, PetscInt stride, PetscInt dual_offset, PetscInt NewNumCellsLocal, Vec *NatOrderVec, IS *NatToPetscIS, Vec *PetscOrderVec) {

  PetscErrorCode ierr;

  // Determine:
  //  - number of cells owned by each rank
  //  - the cell ids (in natural order) owned by each rank after mesh partitioning
  Vec PostPartNatOrderVec;
  ierr = PrePartNatOrder_To_PostPartNatOrder(stride, NewNumCellsLocal, NatToPetscIS, NatOrderVec, &PostPartNatOrderVec); CHKERRQ(ierr);

  // Determine the global cell id offset for each rank after mesh partitioning
  PetscInt NewGlobalOffset = 0;
  ierr = MPI_Exscan(&NewNumCellsLocal, &NewGlobalOffset, 1, MPI_INTEGER, MPI_SUM, PETSC_COMM_WORLD); CHKERRQ(ierr);

  ierr = VecDuplicate(PostPartNatOrderVec, PetscOrderVec); CHKERRQ(ierr);
  ierr = VecCopy(PostPartNatOrderVec, *PetscOrderVec); CHKERRQ(ierr);

  // Save natural ids of local cells owned by each rank after mesh partitioning
  ierr = SaveNaturalCellIDs(ugrid, PostPartNatOrderVec, NewNumCellsLocal, stride); CHKERRQ(ierr);

  // Create application order (AO) from natural-order to PETSc-order
  ierr = CreateApplicationOrder(ugrid, NewGlobalOffset, NewNumCellsLocal); CHKERRQ(ierr);

  // Change cell and dual ids from natural-order to PETSc order
  ierr = CellAndDualIDs_FromNatOrder_To_PetscOrder(ugrid, stride, dual_offset, NewNumCellsLocal, PetscOrderVec);

  // Change the dual ids from PETSc-order to local-order
  ierr = DualIDs_FromPetscOrder_To_LocalOrder(ugrid, stride, dual_offset, NewNumCellsLocal, NewGlobalOffset, PetscOrderVec);

  // Update the array that saves the natural cell ids to include ghost cells
  ierr = UpdateNaturalCellIDs(ugrid, stride, dual_offset, &PostPartNatOrderVec, PetscOrderVec); CHKERRQ(ierr);

  // Determine the ids of cell neigbhors (aka duals) in ghosted-index
  ierr = DetermineNeigbhorsCellIDsInGhostedOrder(ugrid, stride, dual_offset, PetscOrderVec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ScatterVecPetscOrderToLocalOrder(TDyUGrid *ugrid, PetscInt stride, Vec *PetscOrderVec, Vec *LocalOrderVec) {

  PetscErrorCode ierr;

  PetscInt nlmax = ugrid->num_cells_local;
  PetscInt ngmax = ugrid->num_cells_global;
  PetscInt size = ngmax*stride;

  // 1. Create the vector
  ierr = VecCreate(PETSC_COMM_SELF, LocalOrderVec); CHKERRQ(ierr);
  ierr = VecSetSizes(*LocalOrderVec, size, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(*LocalOrderVec, stride); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*LocalOrderVec); CHKERRQ(ierr);

  // 2. Create index sets (ISs)
  PetscInt idx[ngmax];

  // 2.1 Index set to scatter data from a MPI Vec in PETSc-order
  for (PetscInt icell=0; icell<nlmax; icell++) {
    idx[icell] = ugrid->cell_ids_petsc[icell];
  }
  for (PetscInt icell=nlmax; icell<ngmax; icell++) {
    idx[icell] = ugrid->ghost_cell_ids_petsc[icell-nlmax];
  }

  IS is_scatter;
  ierr = ISCreateBlock(PETSC_COMM_WORLD, stride, ngmax, idx, PETSC_COPY_VALUES, &is_scatter); CHKERRQ(ierr);

  // 2.2 Index set to gather data in a Vec in Local-order
  for (PetscInt icell=0; icell<ngmax; icell++) {
    idx[icell] = icell;
  }

  IS is_gather;
  ierr = ISCreateBlock(PETSC_COMM_WORLD, stride, ngmax, idx, PETSC_COPY_VALUES, &is_gather); CHKERRQ(ierr);

  // 3. Create VecScatter
  VecScatter vec_scatter;
  //VecView(*PetscOrderVec,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecScatterCreate(*PetscOrderVec, is_scatter, *LocalOrderVec, is_gather, &vec_scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&is_scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&is_gather); CHKERRQ(ierr);

  // 4. Scatter the data
  ierr = VecScatterBegin(vec_scatter, *PetscOrderVec, *LocalOrderVec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(vec_scatter, *PetscOrderVec, *LocalOrderVec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vec_scatter); CHKERRQ(ierr);

  PetscInt rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  char filename[20];
  sprintf(filename,"elements_local%d.out",rank);
  ierr = TDySavePetscVecAsASCII(*LocalOrderVec, filename); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode ChangeVertexNatOrderToLocalOrder(TDyUGrid *ugrid, PetscInt stride, PetscInt vertex_offset, PetscInt *NewNumVertices, Vec *LocalOrderVec) {

  PetscErrorCode ierr;
  PetscInt rank; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscScalar *v_ptr;
  ierr = VecGetArray(*LocalOrderVec, &v_ptr); CHKERRQ(ierr);

  // Determine the number of vertices
  PetscInt ngmax = ugrid->num_cells_global;
  PetscInt max_nvert = ugrid->max_verts_per_cell;
  PetscInt numVertices=0;

  for (PetscInt icell=0; icell<ngmax; icell++) {
    for (PetscInt ivertex=0; ivertex<max_nvert; ivertex++) {
      PetscInt vertID = (PetscInt) v_ptr[icell*stride + vertex_offset + ivertex];
      if (vertID > 0) {
        numVertices++;
      }
    }
  }

  // Now save the vertex IDs
  PetscInt IntArray1[numVertices];
  numVertices = 0;
  for (PetscInt icell=0; icell<ngmax; icell++) {
    for (PetscInt ivertex=0; ivertex<max_nvert; ivertex++) {
      PetscInt vertID = (PetscInt) v_ptr[icell*stride + vertex_offset + ivertex];
      if (vertID > 0) {
        IntArray1[numVertices++] = vertID;
        v_ptr[icell*stride + vertex_offset + ivertex] = numVertices;
      }
    }
  }
  ierr = VecRestoreArray(*LocalOrderVec, &v_ptr); CHKERRQ(ierr);

  PetscInt IntArray2[numVertices];
  PetscInt IntArray3[numVertices];
  PetscInt IntArray4[numVertices];
  for (PetscInt ivertex=0; ivertex<numVertices; ivertex++) {
    IntArray1[ivertex] = IntArray1[ivertex] - 1;
    IntArray2[ivertex] = ivertex;
    IntArray3[ivertex] = -1;
    IntArray4[ivertex] = -1;
  }

  ierr = PetscSortIntWithPermutation(numVertices, IntArray1, IntArray2); CHKERRQ(ierr);

  PetscInt count=0;
  PetscInt idx=IntArray2[count];
  IntArray3[0] = IntArray1[idx];
  IntArray4[idx] = count;

  for (PetscInt ivertex=0; ivertex<numVertices; ivertex++) {
    PetscInt idx=IntArray2[ivertex];
    PetscInt vertID = IntArray1[idx];
    if (vertID > IntArray3[count]) {
      count++;
      IntArray3[count] = vertID;
    }
    IntArray4[IntArray2[ivertex]] = count;
  }

  numVertices = count+1;
  *NewNumVertices = numVertices;

  // Save vertex ids in natural-order
  ierr = TDyAllocate_IntegerArray_1D(&ugrid->vertex_ids_natural, numVertices); CHKERRQ(ierr);
  for (PetscInt ivertex=0; ivertex<numVertices; ivertex++) {
    ugrid->vertex_ids_natural[ivertex] = IntArray3[ivertex];
  }

  ierr = TDyAllocate_IntegerArray_1D(&ugrid->cell_num_vertices, ngmax); CHKERRQ(ierr);
  ierr = VecGetArray(*LocalOrderVec, &v_ptr); CHKERRQ(ierr);

  PetscInt nlmax = ugrid->num_cells_local;
  ierr = TDyDeallocate_IntegerArray_2D(ugrid->cell_vertices, nlmax); CHKERRQ(ierr);
  ierr = TDyAllocate_IntegerArray_2D(&ugrid->cell_vertices, ngmax, max_nvert); CHKERRQ(ierr);

  for (PetscInt icell=0; icell<ngmax; icell++) {

    PetscInt count=0;
    ugrid->cell_num_vertices[icell] = count;

    for (PetscInt ivertex=0; ivertex<max_nvert; ivertex++) {
      PetscInt vertID = (PetscInt) v_ptr[icell*stride + vertex_offset + ivertex];
      if (vertID > 0) {
        ugrid->cell_vertices[icell][count++] = IntArray4[vertID-1];
        ugrid->cell_num_vertices[icell] = count;
        v_ptr[icell*stride + vertex_offset + ivertex] = IntArray4[vertID-1]+1;
      }
    }
  }

  ierr = VecRestoreArray(*LocalOrderVec, &v_ptr); CHKERRQ(ierr); // Now vertex ids are in local-order

  char filename[30];
  sprintf(filename,"elements_vert_local%d.out",rank);
  ierr = TDySavePetscVecAsASCII(*LocalOrderVec, filename); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode SaveLocalVertexCoordinates(TDyUGrid *ugrid, PetscInt stride, PetscInt vertex_offset, PetscInt NewNumVertices) {

  PetscErrorCode ierr;

  PetscInt dim=3;

  // 1. Create the index set for gathering data in local-order
  IS ISGather;
  PetscInt idx[NewNumVertices];
  for (PetscInt ivertex=0; ivertex<NewNumVertices; ivertex++) {
    idx[ivertex] = ivertex;
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, dim, NewNumVertices, idx, PETSC_COPY_VALUES, &ISGather); CHKERRQ(ierr);
  // 2. Create the index set for scattering data in natural-order
  IS ISScatter;
  for (PetscInt ivertex=0; ivertex<NewNumVertices; ivertex++) {
    idx[ivertex] = ugrid->vertex_ids_natural[ivertex];
  }
  ierr = ISCreateBlock(PETSC_COMM_WORLD, dim, NewNumVertices, idx, PETSC_COPY_VALUES, &ISScatter); CHKERRQ(ierr);

  ierr = TDySavePetscISAsASCII(ISScatter,"is_scatter_vert_old_to_new.out");
  ierr = TDySavePetscISAsASCII(ISGather,"is_gather_vert_old_to_new.out");

  // 3. Create vectors for data in natural-order and local-order
  Vec NatOrderVec;
  ierr = VecCreate(PETSC_COMM_WORLD, &NatOrderVec); CHKERRQ(ierr);
  ierr = VecSetSizes(NatOrderVec, ugrid->num_verts_local*dim, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(NatOrderVec, dim); CHKERRQ(ierr);
  ierr = VecSetFromOptions(NatOrderVec); CHKERRQ(ierr);

  Vec LocalOrderVec;
  ierr = VecCreate(PETSC_COMM_SELF, &LocalOrderVec); CHKERRQ(ierr);
  ierr = VecSetSizes(LocalOrderVec, NewNumVertices*dim, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetBlockSize(LocalOrderVec, dim); CHKERRQ(ierr);
  ierr = VecSetFromOptions(LocalOrderVec); CHKERRQ(ierr);

  // 4. Pack data in natural-ordered vector
  PetscScalar *v_ptr;
  ierr = VecGetArray(NatOrderVec, &v_ptr); CHKERRQ(ierr);
  for (PetscInt ivertex=0; ivertex<ugrid->num_verts_local; ivertex++) {
    for (PetscInt idim=0; idim<dim; idim++) {
      v_ptr[ivertex*dim + idim] = ugrid->vertices[ivertex][idim];
    }
  }
  ierr = VecRestoreArray(NatOrderVec, &v_ptr); CHKERRQ(ierr);

  // 5. Update the memory size for save vertex data in local-order
  ierr = TDyDeallocate_RealArray_2D(ugrid->vertices, ugrid->num_verts_local); CHKERRQ(ierr);
  ugrid->num_verts_natural = ugrid->num_verts_local;
  ugrid->num_verts_local = NewNumVertices;
  ierr = TDyAllocate_RealArray_2D(&ugrid->vertices, ugrid->num_verts_local, dim); CHKERRQ(ierr);

  // 6. Scatter the data
  VecScatter Scatter;
  ierr = VecScatterCreate(NatOrderVec, ISScatter, LocalOrderVec, ISGather, &Scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&ISScatter); CHKERRQ(ierr);
  ierr = ISDestroy(&ISGather); CHKERRQ(ierr);
  ierr = VecScatterBegin(Scatter, NatOrderVec, LocalOrderVec, INSERT_VALUES ,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(Scatter, NatOrderVec, LocalOrderVec, INSERT_VALUES ,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&Scatter); CHKERRQ(ierr);

  // 7. Save the data in data structure
  ierr = VecGetArray(LocalOrderVec, &v_ptr); CHKERRQ(ierr);
  for (PetscInt ivertex=0; ivertex<ugrid->num_verts_local; ivertex++) {
    for (PetscInt idim=0; idim<dim; idim++) {
      ugrid->vertices[ivertex][idim] = v_ptr[ivertex*dim + idim];
    }
  }
  ierr = VecRestoreArray(LocalOrderVec, &v_ptr); CHKERRQ(ierr);
  ierr = VecDestroy(&NatOrderVec); CHKERRQ(ierr);
  ierr = VecDestroy(&LocalOrderVec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------- */
static PetscErrorCode ScatterVertexNatOrderToLocalOrder(TDyUGrid *ugrid, PetscInt stride, PetscInt vertex_offset, Vec *LocalOrderVec) {

  PetscErrorCode ierr;

  PetscInt NewNumVertices=0;
  ierr = ChangeVertexNatOrderToLocalOrder(ugrid, stride, vertex_offset, &NewNumVertices, LocalOrderVec); CHKERRQ(ierr);

  ierr = SaveLocalVertexCoordinates(ugrid, stride, vertex_offset, NewNumVertices); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyUGridCreateFromPFLOTRANMesh(TDyUGrid *ugrid, const char *mesh_file) {

  PetscErrorCode ierr;

  ierr = ReadPFLOTRANMeshFile(mesh_file, ugrid); CHKERRQ(ierr);
  //ierr = UGridPrintCells(ugrid); CHKERRQ(ierr);
  //ierr = UGridPrintVertices(ugrid); CHKERRQ(ierr);

  ierr = DetermineMaxNumVerticesActivePerCell(ugrid); CHKERRQ(ierr);

  Mat AdjMat;
  ierr = CreateAdjacencyMatrix(ugrid, &AdjMat); CHKERRQ(ierr);

  // Create a dual matrix that represents the connectivity between cells.
  // The dual matrix will be partitioning the mesh
  Mat DualMat;
  PetscInt ncommonnodes=3;
  ierr = MatMeshToCellGraph(AdjMat, ncommonnodes, &DualMat); CHKERRQ(ierr);
  ierr = TDySavePetscMatAsASCII(DualMat, "Dual.out"); CHKERRQ(ierr);

  IS NewCellRankIS;
  PetscInt NewNumCellsLocal;
  ierr = PartitionGrid(DualMat, &NewCellRankIS, &NewNumCellsLocal); CHKERRQ(ierr);

  ierr = DetermineMaxNumDualCells(ugrid, DualMat); CHKERRQ(ierr);

  PetscInt vertex_ids_offset = 1 + 1; // +1 for -777
  PetscInt dual_offset = vertex_ids_offset + ugrid->max_verts_per_cell + 1; // +1 for -888
  PetscInt stride = dual_offset + ugrid->max_ndual_per_cell + 1; // +1 for -999999

  IS NatToPetscIS;
  Vec NatOrderVec;
  ierr = CreateISNatOrderToPetscOrder(ugrid, NewCellRankIS, stride, &NatToPetscIS); CHKERRQ(ierr);
  ierr = CreateVectorNatOrder(ugrid, NewCellRankIS, stride, &NatOrderVec); CHKERRQ(ierr);

  ierr = PackNatOrderVector(ugrid, DualMat, &NatOrderVec);
  ierr = MatDestroy(&DualMat); CHKERRQ(ierr);

  Vec PetscOrderVec;
  ierr = ScatterVecNatOrderToPetscOrder(ugrid, stride, dual_offset, NewNumCellsLocal, &NatOrderVec, &NatToPetscIS, &PetscOrderVec);
  ierr = ISDestroy(&NatToPetscIS); CHKERRQ(ierr);

   Vec LocalOrderVec;
   ierr = ScatterVecPetscOrderToLocalOrder(ugrid, stride, &PetscOrderVec, &LocalOrderVec);
   ierr = VecDestroy(&PetscOrderVec); CHKERRQ(ierr);

   ierr = ScatterVertexNatOrderToLocalOrder(ugrid, stride, vertex_ids_offset, &LocalOrderVec);
   ierr = VecDestroy(&LocalOrderVec); CHKERRQ(ierr);

  //ierr = MatDestroy(&AdjMat); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}