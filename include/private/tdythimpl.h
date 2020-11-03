#if !defined(TDYTHIMPL_H)
#define TDYTHIMPL_H

#include <petsc.h>
#include <tdycore.h>

PETSC_INTERN PetscErrorCode TDyTHInitialize(TDy);
PETSC_INTERN PetscErrorCode TDyTHTSPostStep(TS);
PETSC_INTERN PetscErrorCode TDyTHSNESPostCheck(SNESLineSearch,Vec,Vec,Vec,
                                               PetscBool*,PetscBool*,
                                               void*);
PETSC_INTERN PetscErrorCode TDyTHConvergenceTest(SNES,PetscInt,PetscReal,
                                                 PetscReal,PetscReal,
                                                 SNESConvergedReason*,void*);

#endif
