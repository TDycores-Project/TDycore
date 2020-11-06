#if !defined(TDYPERMEABILITYIMPL_H)
#define TDYPERMEABILITYIMPL_H

#include <petsc.h>

typedef enum {
  REL_PERM_FUNC_IRMAY=0,
  REL_PERM_FUNC_MUALEM=1
} TDyRelPermFuncType;

PETSC_INTERN void RelativePermeability_Mualem(PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_INTERN void RelativePermeability_Irmay(PetscReal,PetscReal,PetscReal*,PetscReal*);

#endif
