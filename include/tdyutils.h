#if !defined(TDYUTILS_H)
#define TDYUTILS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscInt GetNumberOfCellVertices(DM dm);
PETSC_EXTERN PetscInt GetNumberOfFaceVertices(DM dm);

#endif
