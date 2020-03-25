#if !defined(TDYMPFAOUTILSIMPL_H)
#define TDYMPFAOUTILSIMPL_H

PETSC_EXTERN PetscErrorCode ExtractSubGmatrix(TDy,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_EXTERN PetscErrorCode ExtractTempSubGmatrix(TDy,PetscInt,PetscInt,PetscInt,PetscReal**);

#endif
