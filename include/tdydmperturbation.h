#if !defined(TDYDMPERTURBATION_H)
#define TDYDMPERTURBATION_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode (*vertexperturbationfunction)(DM,PetscReal);

PETSC_EXTERN PetscErrorCode SetVertexPerturbationFunction(PetscErrorCode(*f)(DM,PetscReal));
PETSC_EXTERN PetscErrorCode PerturbDMVertices(DM,PetscReal);
PETSC_EXTERN PetscErrorCode PerturbDMInteriorVertices(DM,PetscReal);
PETSC_EXTERN PetscErrorCode PerturbVerticesRandom(DM,PetscReal);
PETSC_EXTERN PetscErrorCode PerturbVerticesSmooth(DM,PetscReal);

#endif

