#if !defined(TDYCOREREGIONMPL_H)
#define TDYCOREREGIONMPL_H

#include <petsc.h>

typedef struct {
  PetscInt num_cells;
  PetscInt *cell_ids;
} TDyRegion;

PETSC_INTERN PetscErrorCode TDyRegionCreate(TDyRegion*);
PETSC_INTERN PetscErrorCode TDyRegionAddCells(PetscInt,PetscInt*,TDyRegion*);
PETSC_INTERN PetscErrorCode TDyRegionDestroy(TDyRegion*);
PETSC_INTERN PetscBool TDyRegionAreCellsInTheSameRegion(TDyRegion*,PetscInt,PetscInt);

#endif