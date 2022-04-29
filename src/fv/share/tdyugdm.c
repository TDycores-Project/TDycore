#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <private/tdyugdmimpl.h>
#include <private/tdymemoryimpl.h>

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

static PetscErrorCode ReadPFLOTRANMeshFile(const char *mesh_file, PetscInt ***cell_vertices, PetscReal ***vertices, PetscInt *num_cells_local, PetscInt *max_verts_per_cells, PetscInt *num_verts_local){

  PetscViewer viewer;
  Vec cells, verts;
  PetscScalar *v_p;
  PetscErrorCode ierr;

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
  ierr = VecGetLocalSize(cells, &vec_cells_size);
  ierr = VecGetBlockSize(cells, max_verts_per_cells);
  *num_cells_local = (PetscInt) (vec_cells_size/(*max_verts_per_cells));

  ierr = TDyAllocate_IntegerArray_2D(cell_vertices, *num_cells_local, *max_verts_per_cells); CHKERRQ(ierr);

  ierr = VecGetArray(cells, &v_p); CHKERRQ(ierr);
  PetscInt count=0;
  for (PetscInt i=0; i<*num_cells_local; i++) {
    for (PetscInt j=0; j<*max_verts_per_cells; j++) {
      (*cell_vertices)[i][j] = (PetscInt) v_p[count++];
    }
  }
  ierr = VecRestoreArray(cells, &v_p); CHKERRQ(ierr);
  ierr = VecDestroy(&cells); CHKERRQ(ierr);

  // Save information about vertices
  PetscInt vert_dim, vec_verts_size;
  ierr = VecGetLocalSize(verts, &vec_verts_size);
  ierr = VecGetBlockSize(verts, &vert_dim);
  if (vert_dim!=3) {
    MPI_Comm comm;
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

PetscErrorCode TDyUGDMCreateFromPFLOTRANMesh(TDyUGDM *ugdm, const char *mesh_file) {

  PetscErrorCode ierr;

  PetscInt **cell_vertices;
  PetscReal **vertices;
  PetscInt num_cells_local, max_verts_per_cells, num_verts_local;

  ierr = ReadPFLOTRANMeshFile(mesh_file, &cell_vertices, &vertices, &num_cells_local, &max_verts_per_cells, &num_verts_local); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}
