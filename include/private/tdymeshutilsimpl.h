#if !defined(TDYMESHUTILSIMPL_H)
#define TDYMESHUTILSIMPL_H

PETSC_EXTERN TDyCellType GetCellType(PetscInt, PetscInt);
PETSC_EXTERN PetscInt GetNumVerticesForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumOfCellsSharingAVertexForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumCellsPerEdgeForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumCellsPerFaceForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumOfCellsSharingAFaceForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumOfVerticesFormingAFaceForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumOfEdgesFormingAFaceForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumEdgesForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumNeighborsForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumFacesForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumFacesSharedByVertexForCellType(TDyCellType);
PETSC_EXTERN TDySubcellType GetSubcellTypeForCellType(TDyCellType);
PETSC_EXTERN PetscInt GetNumSubcellsForSubcellType(TDySubcellType);
PETSC_EXTERN PetscInt GetNumOfNuVectorsForSubcellType(TDySubcellType);
PETSC_EXTERN PetscInt GetNumVerticesForSubcellType(TDySubcellType);
PETSC_EXTERN PetscInt GetNumFacesForSubcellType(TDySubcellType);
PETSC_EXTERN PetscInt TDyMeshGetNumberOfLocalCells(TDy_mesh*);
PETSC_EXTERN PetscInt TDyMeshGetNumberOfLocalFacess(TDy_mesh*);
PETSC_EXTERN PetscInt TDyMeshGetNumberOfNonLocalFacess(TDy_mesh*);
PETSC_EXTERN PetscInt TDyMeshGetNumberOfNonInternalFacess(TDy_mesh*);
PETSC_EXTERN PetscErrorCode AreFacesNeighbors(TDy_face*, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TDySubCell_GetIthNuVector(TDy_subcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDySubCell_GetIthFaceCentroid(TDy_subcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDy_subcell*, PetscInt, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyEdge_GetCentroid(TDy_edge*, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyEdge_GetCentroid(TDy_edge*, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyFace_GetCentroid(TDy_face*, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyFace_GetNormal(TDy_face*, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode TDyVertex_GetCoordinate(TDy_vertex*, PetscInt, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode TDyCell_GetCentroid2(TDy_cell*, PetscInt, PetscInt, PetscReal*);
PETSC_EXTERN PetscErrorCode FindNeighboringVerticesOfAFace(TDy_face*, PetscInt, PetscInt, PetscInt[2]);
PETSC_EXTERN PetscErrorCode SetupCell2CellConnectivity(TDy_vertex*, PetscInt, TDy_cell*, TDy_face*, TDy_subcell*, PetscInt**);
PETSC_EXTERN PetscErrorCode FindFaceIDsOfACellCommonToAVertex(PetscInt, TDy_face*, TDy_vertex*, PetscInt,PetscInt[3],PetscInt*);
PETSC_EXTERN PetscErrorCode IdentifyLocalCells(TDy);
PETSC_EXTERN PetscErrorCode IdentifyLocalVertices(TDy);
PETSC_EXTERN PetscErrorCode IdentifyLocalEdges(TDy);
PETSC_EXTERN PetscErrorCode IdentifyLocalFaces(TDy);
PETSC_EXTERN PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDy_cell*, PetscInt, TDy_vertex*, PetscInt, TDy_subcell*, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyGetNumCellsLocal(TDy, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyGetCellNaturalIDsLocal(TDy, PetscInt*, PetscInt[]);
PETSC_EXTERN PetscErrorCode TDyGetCellIsLocal(TDy, PetscInt*, PetscInt[]);
#endif
