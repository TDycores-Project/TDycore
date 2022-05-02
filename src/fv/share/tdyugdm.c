#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <private/tdyugdmimpl.h>
#include <private/tdymemoryimpl.h>

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
  PetscInt *max_verts_per_cells = &ugrid->max_verts_per_cells;
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

  *max_verts_per_cells = blocksize - 1;
  ierr = TDyAllocate_IntegerArray_2D(cell_vertices, *num_cells_local, *max_verts_per_cells); CHKERRQ(ierr);

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
    for (PetscInt j=1; j<*max_verts_per_cells+1; j++) {
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
        for (PetscInt ivertex=0; ivertex<ugrid->max_verts_per_cells; ivertex++) {
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
    for (PetscInt ivertex=0; ivertex<ugrid->max_verts_per_cells; ivertex++) {
      if (ugrid->cell_vertices[icell][ivertex] > 0) {
        tmp++;
      }
    }
    if (tmp > nvert_active){
      nvert_active = tmp;
    }
  }

  ierr = MPI_Allreduce(&nvert_active, &ugrid->max_nvert_active_per_cell, 1, MPI_INTEGER, MPI_MAX, PETSC_COMM_WORLD); CHKERRQ(ierr);
  printf("max_nvert_active_per_cell = %d\n",ugrid->max_nvert_active_per_cell);

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

  PetscFunctionReturn(0);

}
