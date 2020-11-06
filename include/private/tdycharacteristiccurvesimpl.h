#if !defined(TDYCHARACTERISTICCURVESIMPL_H)
#define TDYCHARACTERISTICCURVESIMPL_H

#include <petsc.h>

typedef enum {
  REL_PERM_FUNC_IRMAY=0,
  REL_PERM_FUNC_MUALEM=1
} TDyRelPermFuncType;

PETSC_INTERN void RelativePermeability_Mualem(PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_INTERN void RelativePermeability_Irmay(PetscReal,PetscReal,PetscReal*,PetscReal*);

#endif
