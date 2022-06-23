#include <private/tdyregionimpl.h>
#include <private/tdymemoryimpl.h>

/// Initializes a region to which mesh cells can be added. The region contains
/// no cells.
/// @param [out] region A pointer to a TDyRegion struct to be initialized. No
///                     memory is allocated by this function.
/// @returns 0 on success, or a non-zero error code on failure.
PetscErrorCode TDyRegionCreate(TDyRegion *region) {

  region->num_cells = 0;
  region->cell_ids = NULL;

  PetscFunctionReturn(0);
}

/// Adds a set of cells to an existing region.
/// @param [in] ncells A positive number indicating how many cells are added to
///                    the region.
/// @param [in] ids An array of cell indexes of length ncells. These indexes are
///                 copied to the region.
/// @param [inout] region A pointer to a TDyRegion struct. Memory is allocated as
///                       needed to accommodate cells.
/// @returns 0 on success, or a non-zero error code on failure.
PetscErrorCode TDyRegionAddCells(PetscInt ncells, PetscInt ids[], TDyRegion *region) {

  PetscErrorCode ierr;
  PetscInt ii;

  region->num_cells = ncells;
  ierr = TDyAllocate_IntegerArray_1D(&region->cell_ids, ncells); CHKERRQ(ierr);
  for (ii=0; ii<ncells; ii++) region->cell_ids[ii] = ids[ii];

  PetscFunctionReturn(0);
}

/// Destroys the given region, freeing any resources allocated to it‥
/// @param [out] region A pointer to a region struct to be destroyed.
/// @returns 0 on success, or a non-zero error code on failure.
PetscErrorCode TDyRegionDestroy(TDyRegion *region) {
  if (region->cell_ids != NULL) {
    return TDyFree(region->cell_ids);
  }
  PetscFunctionReturn(0);
}

/// Returns true if the cells at the given indexes within the region belong to
/// the same region, false otherwise.
/// @param [in] region A pointer to a region struct to be destroyed.
/// @param [in] cell_id_1 The index for a cell within the given region.
/// @param [in] cell_id_2 The index for a second cell within the given region.
PetscBool TDyRegionAreCellsInTheSameRegion(TDyRegion *region, PetscInt cell_id_1, PetscInt cell_id_2) {

  if (cell_id_1 >= region->num_cells || cell_id_2 >= region->num_cells) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Cell ID is larger than the number of cells in the region");
  }

  PetscFunctionReturn (region->cell_ids[cell_id_1] == region->cell_ids[cell_id_2]);
}
