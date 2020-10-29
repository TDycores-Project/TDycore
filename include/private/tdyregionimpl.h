#if !defined(TDYCOREREGIONMPL_H)
#define TDYCOREREGIONMPL_H

#include <petsc.h>

typedef struct _TDyRegion TDyRegion;

struct _TDyRegion {
  PetscInt num_cells;
  PetscInt *id;
};

#endif