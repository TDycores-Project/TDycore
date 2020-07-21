#if !defined(TDYDM_H)
#define TDYDM_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode TDyCreateDM(DM*);
PETSC_EXTERN PetscErrorCode TDyDistributeDM(DM*);

#endif

