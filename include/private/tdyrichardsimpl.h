#if !defined(TDYRICHARDSIMPL_H)
#define TDYRICHARDSIMPL_H

#include <petsc.h>
#include <tdycore.h>

PETSC_INTERN PetscErrorCode TDyRichardsInitialize(TDy); // TDyDriver setup fn
PETSC_INTERN PetscErrorCode TDyRichardsTSPostStep(TS);
PETSC_INTERN PetscErrorCode TDyRichardsSNESPostCheck(SNESLineSearch,Vec,Vec,Vec,
                                                     PetscBool*,PetscBool*,
                                                     void*);
PETSC_INTERN PetscErrorCode TDyRichardsConvergenceTest(SNES,PetscInt,PetscReal,
                                                  PetscReal,PetscReal,
                                                  SNESConvergedReason*,void*);

#endif
