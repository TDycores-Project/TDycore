#if !defined(TDYSALINITYIMPL_H)
#define TDYSALINITYIMPL_H

#include <petsc.h>
#include <tdycore.h>

PETSC_INTERN PetscErrorCode TDySalinityInitialize(TDy);
PETSC_INTERN PetscErrorCode TDySalinityTSPostStep(TS);
PETSC_INTERN PetscErrorCode TDySalinitySNESPostCheck(SNESLineSearch,Vec,Vec,Vec,
                                                     PetscBool*,PetscBool*,
                                                     void*);
PETSC_INTERN PetscErrorCode TDySalinityConvergenceTest(SNES,PetscInt,PetscReal,
                                                       PetscReal,PetscReal,
                                                       SNESConvergedReason*,void*);

#endif
