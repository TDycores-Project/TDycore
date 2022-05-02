#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <private/tdyugdmimpl.h>
#include <private/tdymemoryimpl.h>
#include <private/tdyutils.h>

static PetscInt vertex_separator = -777;
static PetscInt dual_separator = -888;
static PetscInt cell_separator = -999999;

/* ---------------------------------------------------------------- */
PetscErrorCode TDyUGDMCreate(TDyUGDM *ugdm){
  ugdm = malloc(sizeof(TDyUGDM));

  ugdm->IS_GhostedCells_in_LocalOrder = NULL;
  ugdm->IS_GhostedCells_in_PetscOrder = NULL;

  ugdm->IS_LocalCells_in_LocalOrder = NULL;
  ugdm->IS_LocalCells_in_PetscOrder = NULL;

  ugdm->IS_GhostCells_in_LocalOrder = NULL;
  ugdm->IS_GhostCells_in_PetscOrder = NULL;

  ugdm->Scatter_LocalCells_to_GlobalCells = NULL;
  ugdm->Scatter_LocalCells_to_LocalCells = NULL;
  ugdm->Scatter_LocalCells_to_LocalCells = NULL;
  ugdm->Scatter_GlobalCells_to_NaturalCells = NULL;

  ugdm->Mapping_LocalCells_to_NaturalCells = NULL;

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
  printf("max_verts_per_cell = %d\n",ugrid->max_verts_per_cell);

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

  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"is.out",&viewer); CHKERRQ(ierr);
  ierr = ISView(*NewCellRankIS, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

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
static PetscErrorCode CreatePrePartitionVector(TDyUGrid *ugrid, IS NewCellRankIS, PetscInt stride, IS *OldToNewIS, Vec *OldVec) {

  PetscErrorCode ierr;

  IS NumberingIS;
  const PetscInt *is_ptr;

  ierr = ISPartitioningToNumbering(NewCellRankIS, &NumberingIS); CHKERRQ(ierr);
  ierr = ISGetIndices(NumberingIS, &is_ptr); CHKERRQ(ierr);

  PetscInt num_cells_local_old = ugrid->num_cells_local;
  ierr = ISCreateBlock(PETSC_COMM_WORLD, stride, num_cells_local_old, is_ptr, PETSC_COPY_VALUES, OldToNewIS); CHKERRQ(ierr);

  ierr = ISRestoreIndices(NumberingIS, &is_ptr); CHKERRQ(ierr);
  ierr = ISDestroy(&NumberingIS); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, OldVec); CHKERRQ(ierr);
  ierr = VecSetSizes(*OldVec, stride*num_cells_local_old, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*OldVec); CHKERRQ(ierr);

  PetscViewer viewer;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"is_scatter_elem_old_to_new.out",&viewer); CHKERRQ(ierr);
  ierr = ISView(*OldToNewIS, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
static PetscErrorCode PackPrePartitionVector(TDyUGrid *ugrid, Mat DualMat, Vec *OldVec) {

  PetscErrorCode ierr;

  const PetscInt *ia_ptr, *ja_ptr;
  PetscBool success;
  PetscInt num_rows;
  ierr = MatGetRowIJ(DualMat, 0, PETSC_FALSE, PETSC_FALSE, &num_rows, &ia_ptr, &ja_ptr, &success);CHKERRQ(ierr);

  PetscScalar *v_ptr;
  ierr = VecGetArray(*OldVec, &v_ptr); CHKERRQ(ierr);

  PetscInt count=0, vertex_count=0;
  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt global_offset = 0;
  ierr = MPI_Exscan(&ugrid->num_cells_local, &global_offset, 1, MPI_INTEGER, MPI_SUM, PETSC_COMM_WORLD); CHKERRQ(ierr);

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

  ierr = VecRestoreArray(*OldVec, &v_ptr); CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(DualMat, 0, PETSC_FALSE, PETSC_FALSE, &num_rows, &ia_ptr, &ja_ptr, &success);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode PrePartNatOrder_To_PostPartNatOrder(PetscInt stride, PetscInt NewNumCellsLocal, IS *OldToNewIS, Vec *Pre, Vec *Post) {

  PetscErrorCode ierr;

  ierr = VecCreate(PETSC_COMM_WORLD, Post); CHKERRQ(ierr);
  ierr = VecSetSizes(*Post, stride*NewNumCellsLocal, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*Post); CHKERRQ(ierr);

  VecScatter VecScatter;
  ierr = VecScatterCreate(*Pre, PETSC_NULL, *Post, *OldToNewIS, &VecScatter); CHKERRQ(ierr);
  ierr = VecScatterBegin(VecScatter, *Pre, *Post, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(VecScatter, *Pre, *Post, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&VecScatter); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
  PetscErrorCode ScatterVecPrePartitionToPETScOrder(TDyUGrid *ugrid, PetscInt stride, PetscInt dual_offset, PetscInt NewNumCellsLocal, Vec *OldVec, IS *OldToNewIS, Vec *PetscOrderVec) {

    PetscErrorCode ierr;

    Vec PostPartNatOrderVec;
    ierr = PrePartNatOrder_To_PostPartNatOrder(stride, NewNumCellsLocal, OldToNewIS, OldVec, &PostPartNatOrderVec); CHKERRQ(ierr);
    ierr = TDySavePetscVecAsASCII(*OldVec,"elements_old.out");
    ierr = TDySavePetscVecAsASCII(PostPartNatOrderVec,"elements_natural.out");

    PetscFunctionReturn(0);
  }

/* ---------------------------------------------------------------- */
PetscErrorCode TDyUGDMCreateFromPFLOTRANMesh(TDyUGDM *ugdm, const char *mesh_file) {

  PetscErrorCode ierr;
  TDyUGrid ugrid;

  ierr = ReadPFLOTRANMeshFile(mesh_file, &ugrid); CHKERRQ(ierr);
  ierr = UGridPrintCells(&ugrid); CHKERRQ(ierr);
  ierr = UGridPrintVertices(&ugrid); CHKERRQ(ierr);

  ierr = DetermineMaxNumVerticesActivePerCell(&ugrid); CHKERRQ(ierr);

  Mat AdjMat;
  ierr = CreateAdjacencyMatrix(&ugrid, &AdjMat); CHKERRQ(ierr);

  // Create a dual matrix that represents the connectivity between cells.
  // The dual matrix will be partitioning the mesh
  Mat DualMat;
  PetscInt ncommonnodes=3;
  ierr = MatMeshToCellGraph(AdjMat, ncommonnodes, &DualMat); CHKERRQ(ierr);
  ierr = TDySavePetscMatAsASCII(DualMat, "Dual.out"); CHKERRQ(ierr);

  IS NewCellRankIS;
  PetscInt NewNumCellsLocal;
  ierr = PartitionGrid(DualMat, &NewCellRankIS, &NewNumCellsLocal); CHKERRQ(ierr);

  ierr = DetermineMaxNumDualCells(&ugrid, DualMat); CHKERRQ(ierr);

  PetscInt vertex_ids_offset = 1 + 1; // +1 for -777
  PetscInt dual_offset = vertex_ids_offset + ugrid.max_verts_per_cell + 1; // +1 for -888
  PetscInt stride = dual_offset + ugrid.max_ndual_per_cell + 1; // +1 for -999999

  IS OldToNewIS;
  Vec OldVec;
  ierr = CreatePrePartitionVector(&ugrid, NewCellRankIS, stride, &OldToNewIS, &OldVec); CHKERRQ(ierr);

  ierr = PackPrePartitionVector(&ugrid, DualMat, &OldVec);
  ierr = MatDestroy(&DualMat); CHKERRQ(ierr);

  Vec PetscOrderVec;
  ierr = ScatterVecPrePartitionToPETScOrder(&ugrid, stride, dual_offset, NewNumCellsLocal, &OldVec, &OldToNewIS, &PetscOrderVec);
  ierr = ISDestroy(&OldToNewIS); CHKERRQ(ierr);

  //ierr = MatDestroy(&AdjMat); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
