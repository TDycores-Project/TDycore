#if !defined(TDYCOREREGIONMPL_H)
#define TDYCOREREGIONMPL_H

#include <petsc.h>

typedef struct _TDyRegion TDyRegion;

struct _TDyRegion {
  PetscInt num_cells;
  PetscInt *id;
};

PETSC_INTERN PetscErrorCode TDyRegionCreate(TDyRegion*);
PETSC_INTERN PetscErrorCode TDyRegionAddCells(PetscInt,PetscInt*,TDyRegion*);
PETSC_INTERN PetscErrorCode TDyRegionDestroy(TDyRegion*);

#endif