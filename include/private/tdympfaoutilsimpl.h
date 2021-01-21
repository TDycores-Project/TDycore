#if !defined(TDYMPFAOUTILSIMPL_H)
#define TDYMPFAOUTILSIMPL_H

PETSC_INTERN PetscErrorCode ExtractSubGmatrix(TDy,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode ExtractTempSubGmatrix(TDy,PetscInt,PetscInt,PetscInt,PetscReal**);

#endif
