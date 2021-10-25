#if !defined(TDYFEIMPL_H)
#define TDYFEIMPL_H

#include <petsc.h>
#include <tdycore.h>

// Functions common to multiple finite element implementations.
PETSC_INTERN PetscErrorCode TDyCreateCellVertexMap(TDy,PetscInt**);
PETSC_INTERN PetscErrorCode TDyCreateCellVertexDirFaceMap(TDy,PetscInt**);
PETSC_INTERN PetscErrorCode TDyQuadrature(PetscQuadrature,PetscInt);
PETSC_INTERN void HdivBasisQuad(const PetscReal*,PetscReal*,PetscReal*,PetscReal);
PETSC_INTERN void HdivBasisHex(const PetscReal*,PetscReal*,PetscReal*,PetscReal);

#endif
