#include <private/tdyregionimpl.h>
#include <private/tdymemoryimpl.h>

PetscErrorCode TDyRegionCreate(TDyRegion *region) {

  region->num_cells = 0;
  region->id = NULL;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegionAddCells(PetscInt ncells, PetscInt *id, TDyRegion *region) {

  PetscErrorCode ierr;

  region->num_cells = ncells;
  ierr = TDyAllocate_IntegerArray_1D(&region->id, ncells); CHKERRQ(ierr);
  ierr = PetscMemcpy(region->id, id, ncells); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegionDestroy(TDyRegion *region) {
  if (region->id != NULL) {
    return PetscFree(region->id);
  }
  PetscFunctionReturn(0);
}
