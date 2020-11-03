#if !defined(TDYDMIMPL_H)
#define TDYDMIMPL_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode TDyCreateDM(DM*);
PETSC_EXTERN PetscErrorCode TDyDistributeDM(DM*);

#endif

