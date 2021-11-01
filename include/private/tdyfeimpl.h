#if !defined(TDYFEIMPL_H)
#define TDYFEIMPL_H

#include <petsc.h>
#include <tdycore.h>

// Functions common to multiple finite element implementations.
PETSC_INTERN PetscErrorCode CreateCellVertexMap(DM,PetscInt,PetscReal*,PetscInt**);
PETSC_INTERN PetscErrorCode CreateCellVertexDirFaceMap(DM,PetscInt,PetscReal*,PetscReal*,PetscInt*,PetscInt**);
PETSC_INTERN PetscErrorCode SetQuadrature(PetscQuadrature,PetscInt);
PETSC_INTERN void HdivBasisQuad(const PetscReal*,PetscReal*,PetscReal*,PetscReal);
PETSC_INTERN void HdivBasisHex(const PetscReal*,PetscReal*,PetscReal*,PetscReal);

#endif
