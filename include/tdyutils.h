#if !defined(TDYUTILS_H)
#define TDYUTILS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscInt GetNumberOfCellVertices(DM dm);
PETSC_EXTERN PetscInt GetNumberOfFaceVertices(DM dm);
PETSC_EXTERN PetscErrorCode ComputeLength(PetscReal v1[3], PetscReal v2[3], PetscInt dim, PetscReal *length);

#endif
