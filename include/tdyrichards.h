#if !defined(TDYRICHARDS_H)
#define TDYRICHARDS_H

#include <petsc.h>
#include <tdycore.h>

PETSC_EXTERN PetscErrorCode TDyRichardsInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyRichardsTSPostStep(TS);
PETSC_EXTERN PetscErrorCode TDyRichardsSNESPostCheck(SNESLineSearch,Vec,Vec,Vec,
                                                     PetscBool*,PetscBool*,
                                                     void*);
PETSC_EXTERN PetscErrorCode TDyRichardsConvergenceTest(SNES,PetscInt,PetscReal,
                                                  PetscReal,PetscReal,
                                                  SNESConvergedReason*,void*);

#endif
