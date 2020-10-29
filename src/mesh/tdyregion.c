#include <private/tdyregionimpl.h>
#include <private/tdymemoryimpl.h>

PetscErrorCode TDyRegionCreate(TDyRegion *region) {

  region->num_cells = 0;
  region->cell_ids = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegionAddCells(PetscInt ncells, PetscInt *ids, TDyRegion *region) {

  PetscErrorCode ierr;

  region->num_cells = ncells;
  ierr = TDyAllocate_IntegerArray_1D(&region->cell_ids, ncells); CHKERRQ(ierr);
  ierr = PetscMemcpy(region->cell_ids, ids, ncells); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegionDestroy(TDyRegion *region) {
  if (region->cell_ids != NULL) {
    return PetscFree(region->cell_ids);
  }
  PetscFunctionReturn(0);
}
