#if !defined(TDYUTILS_H)
#define TDYUTILS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscErrorCode TDySaveClosures(DM,PetscInt*,PetscInt**,PetscInt);
PETSC_EXTERN PetscInt GetNumberOfCellVertices(DM,PetscInt*,PetscInt**);
PETSC_EXTERN PetscInt GetNumberOfFaceVertices(DM dm);
PETSC_EXTERN PetscErrorCode ComputeLength(PetscReal v1[3], PetscReal v2[3], PetscInt dim, PetscReal *length);
PETSC_EXTERN PetscErrorCode CrossProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal cross_P[3]);
PETSC_EXTERN PetscErrorCode DotProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal *dot_P);
PETSC_EXTERN PetscErrorCode TriangleArea(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal *area);
PETSC_EXTERN PetscErrorCode QuadrilateralArea(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3], PetscReal *area);
PETSC_EXTERN PetscErrorCode NormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3],PetscReal normal[3]);
PETSC_EXTERN PetscErrorCode UnitNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3],PetscReal normal[3]);
PETSC_EXTERN PetscErrorCode NormalToQuadrilateral(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3],PetscReal normal[3]);
PETSC_EXTERN PetscErrorCode ComputeVolumeOfTetrahedron(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3],PetscReal *volume);
PETSC_EXTERN PetscErrorCode CreateVecJoiningTwoVertices(PetscReal vtx_from[3],PetscReal vtx_to[3], PetscReal vec[3]);
PETSC_EXTERN PetscInt ReturnIndexInList(PetscInt *list, PetscInt nlist, PetscInt value);
PETSC_EXTERN PetscInt SavePetscVecAsBinary(Vec vec, const char filename[]);

#endif
