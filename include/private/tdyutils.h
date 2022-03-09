#if !defined(TDYUTILS_H)
#define TDYUTILS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

PETSC_INTERN PetscErrorCode TDySaveClosures(DM,PetscInt*,PetscInt**,PetscInt*);
PETSC_INTERN PetscInt TDyGetNumberOfCellVerticesWithClosures(DM,PetscInt*,PetscInt**);
PETSC_INTERN PetscInt TDyMaxNumberOfCellsSharingAVertex(DM,PetscInt*, PetscInt**);
PETSC_INTERN PetscInt TDyMaxNumberOfFacesSharingAVertex(DM,PetscInt*, PetscInt**);
PETSC_INTERN PetscInt TDyMaxNumberOfEdgesSharingAVertex(DM,PetscInt*, PetscInt**);
PETSC_INTERN PetscErrorCode TDyComputeLength(PetscReal v1[3], PetscReal v2[3], PetscInt dim, PetscReal *length);
PETSC_INTERN PetscErrorCode TDyCrossProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal cross_P[3]);
PETSC_INTERN PetscErrorCode TDyDotProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal *dot_P);
PETSC_INTERN PetscErrorCode TDyTriangleArea(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal *area);
PETSC_INTERN PetscErrorCode TDyQuadrilateralArea(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3], PetscReal *area);
PETSC_INTERN PetscErrorCode TDyUnitNormalVectorJoiningTwoVertices(PetscReal[3], PetscReal[3],PetscReal[3]);
PETSC_INTERN PetscErrorCode TDyNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3],PetscReal normal[3]);
PETSC_INTERN PetscErrorCode TDyUnitNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3],PetscReal normal[3]);
PETSC_INTERN PetscErrorCode TDyNormalToQuadrilateral(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3],PetscReal normal[3]);
PETSC_INTERN PetscErrorCode TDyComputeVolumeOfTetrahedron(PetscReal node_1[3], PetscReal node_2[3],PetscReal node_3[3], PetscReal node_4[3],PetscReal *volume);
PETSC_INTERN PetscErrorCode TDyCreateVecJoiningTwoVertices(PetscReal vtx_from[3],PetscReal vtx_to[3], PetscReal vec[3]);
PETSC_INTERN PetscInt TDyReturnIndexInList(PetscInt *list, PetscInt nlist, PetscInt value);
PETSC_INTERN PetscInt TDySavePetscVecAsBinary(Vec vec, const char filename[]);
PETSC_INTERN PetscInt TDyReadBinaryPetscVec(Vec vec, MPI_Comm comm, const char filename[]);
PETSC_INTERN PetscInt TDySavePetscMatAsBinary(Mat, const char []);
PETSC_INTERN PetscErrorCode ExtractSubVectors(Vec,PetscInt,Vec *);
PETSC_INTERN PetscErrorCode ComputeTheta(PetscReal, PetscReal, PetscReal *);
PETSC_INTERN PetscErrorCode ComputeDeterminantOf3by3Matrix(PetscReal [9], PetscReal *);
PETSC_INTERN PetscErrorCode ComputeInverseOf3by3Matrix(PetscReal[9], PetscReal[9]);
PETSC_INTERN PetscErrorCode ComputePlaneGeometry (PetscReal[3], PetscReal[3], PetscReal[3], PetscReal[4]);
PETSC_INTERN PetscErrorCode GeometryGetPlaneIntercept (PetscReal[4], PetscReal[3], PetscReal[3], PetscReal[4]);
#endif
