#if !defined(TDYRICHARDS_H)
#define TDYRICHARDS_H

#include <petsc.h>
#include <tdycore.h>

PETSC_INTERN PetscErrorCode TDyRichardsInitialize(TDy);
PETSC_INTERN PetscErrorCode TDyRichardsTSPostStep(TS);
PETSC_INTERN PetscErrorCode TDyRichardsSNESPostCheck(SNESLineSearch,Vec,Vec,Vec,
                                                     PetscBool*,PetscBool*,
                                                     void*);
PETSC_INTERN PetscErrorCode TDyRichardsConvergenceTest(SNES,PetscInt,PetscReal,
                                                  PetscReal,PetscReal,
                                                  SNESConvergedReason*,void*);

#endif
