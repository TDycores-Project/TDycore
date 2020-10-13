#if !defined(TDYTH_H)
#define TDYTH_H

#include <petsc.h>
#include <tdycore.h>

PETSC_EXTERN PetscErrorCode TDyTHInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyTHTSPostStep(TS);
PETSC_EXTERN PetscErrorCode TDyTHSNESPostCheck(SNESLineSearch,Vec,Vec,Vec,
                                               PetscBool*,PetscBool*,
                                               void*);
PETSC_EXTERN PetscErrorCode TDyTHConvergenceTest(SNES,PetscInt,PetscReal,
                                                 PetscReal,PetscReal,
                                                 SNESConvergedReason*,void*);

#endif
