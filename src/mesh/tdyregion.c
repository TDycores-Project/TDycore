#include <private/tdyregionimpl.h>
#include <private/tdymemoryimpl.h>

PetscErrorCode TDyRegionCreate(TDyRegion *region) {

  region->num_cells = 0;
  region->cell_ids = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegionAddCells(PetscInt ncells, PetscInt *ids, TDyRegion *region) {

  PetscErrorCode ierr;
  PetscInt ii;

  region->num_cells = ncells;
  ierr = TDyAllocate_IntegerArray_1D(&region->cell_ids, ncells); CHKERRQ(ierr);
  for (ii=0; ii<ncells; ii++) region->cell_ids[ii] = ids[ii];

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegionDestroy(TDyRegion *region) {
  if (region->cell_ids != NULL) {
    return PetscFree(region->cell_ids);
  }
  PetscFunctionReturn(0);
}

PetscBool TDyRegionAreCellsInTheSameRegion(TDyRegion *region, PetscInt cell_id_1, PetscInt cell_id_2) {

  if (cell_id_1 >= region->num_cells || cell_id_2 >= region->num_cells) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Cell ID is larger than the number of cells in the region");
  }

  PetscFunctionReturn (region->cell_ids[cell_id_1] == region->cell_ids[cell_id_2]);
}
