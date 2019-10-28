#if !defined(TDYSATURATIONIMPL_H)
#define TDYSATURATIONIMPL_H

#include <petsc.h>

typedef enum {
  SAT_FUNC_GARDNER=0,
  SAT_FUNC_VAN_GENUCHTEN=1
} TDySatFuncType;

PETSC_EXTERN void PressureSaturation_VanGenuchten(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void PressureSaturation_Gardner(PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);

#endif
