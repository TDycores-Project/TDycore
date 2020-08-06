#if !defined(TDYUTILS_H)
#define TDYUTILS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscErrorCode TDySaveClosures(DM,PetscInt*,PetscInt**,PetscInt*);
PETSC_EXTERN PetscInt TDyGetNumberOfCellVerticesWithClosures(DM,PetscInt*,PetscInt**);
PETSC_EXTERN PetscInt TDyMaxNumberOfCellsSharingAVertex(DM,PetscInt*, PetscInt**);
PETSC_EXTERN PetscInt TDyMaxNumberOfFacesSharingAVertex(DM,PetscInt*, PetscInt**);
PETSC_EXTERN PetscInt TDyMaxNumberOfEdgesSharingAVertex(DM,PetscInt*, PetscInt**);
PETSC_EXTERN PetscErrorCode TDyComputeLength(PetscReal v1[3], PetscReal v2[3], PetscInt dim, PetscReal *length);
PETSC_EXTERN PetscErrorCode TDyCrossProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal cross_P[3]);
PETSC_EXTERN PetscErrorCode TDyDotProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal *dot_P);
PETSC_EXTERN PetscErrorCode TDyTriangleArea(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal *area);
PETSC_EXTERN PetscErrorCode TDyQuadrilateralArea(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3], PetscReal *area);
PETSC_EXTERN PetscErrorCode TDyUnitNormalVectorJoiningTwoVertices(PetscReal[3], PetscReal[3],PetscReal[3]);
PETSC_EXTERN PetscErrorCode TDyNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3],PetscReal normal[3]);
PETSC_EXTERN PetscErrorCode TDyUnitNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3],PetscReal normal[3]);
PETSC_EXTERN PetscErrorCode TDyNormalToQuadrilateral(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3],PetscReal normal[3]);
PETSC_EXTERN PetscErrorCode TDyComputeVolumeOfTetrahedron(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3],PetscReal *volume);
PETSC_EXTERN PetscErrorCode TDyCreateVecJoiningTwoVertices(PetscReal vtx_from[3],PetscReal vtx_to[3], PetscReal vec[3]);
PETSC_EXTERN PetscInt TDyReturnIndexInList(PetscInt *list, PetscInt nlist, PetscInt value);
PETSC_EXTERN PetscInt TDySavePetscVecAsBinary(Vec vec, const char filename[]);
PETSC_EXTERN PetscErrorCode ExtractSubVectors(Vec,PetscInt,Vec *);
PETSC_EXTERN PetscErrorCode ComputeTheta(PetscReal, PetscReal, PetscReal *);
#endif
