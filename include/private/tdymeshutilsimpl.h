#if !defined(TDYMESHUTILSIMPL_H)
#define TDYMESHUTILSIMPL_H

PETSC_INTERN TDyCellType GetCellType(PetscInt, PetscInt);
PETSC_INTERN PetscInt GetNumVerticesForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfCellsSharingAVertexForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumCellsPerEdgeForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumCellsPerFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfCellsSharingAFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfVerticesFormingAFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumOfEdgesFormingAFaceForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumEdgesForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumNeighborsForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumFacesForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumFacesSharedByVertexForCellType(TDyCellType);
PETSC_INTERN TDySubcellType GetSubcellTypeForCellType(TDyCellType);
PETSC_INTERN PetscInt GetNumSubcellsForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt GetNumOfNuVectorsForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt GetNumVerticesForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt GetNumFacesForSubcellType(TDySubcellType);
PETSC_INTERN PetscInt TDyMeshGetNumberOfLocalCells(TDyMesh*);
PETSC_INTERN PetscInt TDyMeshGetNumberOfLocalFacess(TDyMesh*);
PETSC_INTERN PetscInt TDyMeshGetNumberOfNonLocalFacess(TDyMesh*);
PETSC_INTERN PetscInt TDyMeshGetNumberOfNonInternalFacess(TDyMesh*);
PETSC_INTERN PetscErrorCode AreFacesNeighbors(TDyFace*, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthNuVector(TDySubcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthFaceCentroid(TDySubcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDySubcell*, PetscInt, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyEdge_GetCentroid(TDyEdge*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyFace_GetCentroid(TDyFace*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyFace_GetNormal(TDyFace*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyVertex_GetCoordinate(TDyVertex*, PetscInt, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode TDyCell_GetCentroid2(TDyCell*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode FindNeighboringVerticesOfAFace(TDyFace*, PetscInt, PetscInt, PetscInt[2]);
PETSC_INTERN PetscErrorCode SetupCell2CellConnectivity(TDyVertex*, PetscInt, TDyCell*, TDyFace*, TDySubcell*, PetscInt**);
PETSC_INTERN PetscErrorCode FindFaceIDsOfACellCommonToAVertex(PetscInt, TDyFace*, TDyVertex*, PetscInt,PetscInt[3],PetscInt*);
PETSC_INTERN PetscErrorCode IdentifyLocalCells(TDy);
PETSC_INTERN PetscErrorCode IdentifyLocalVertices(TDy);
PETSC_INTERN PetscErrorCode IdentifyLocalEdges(TDy);
PETSC_INTERN PetscErrorCode IdentifyLocalFaces(TDy);
PETSC_INTERN PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDyCell*, PetscInt, TDyVertex*, PetscInt, TDySubcell*, PetscInt*);
PETSC_INTERN PetscErrorCode TDyPrintSubcellInfo(TDy, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode TDyPrintFaceInfo(TDy, PetscInt);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthNuStarVector(TDySubcell*,PetscInt,PetscInt,PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyEdge_GetNormal(TDyEdge*,PetscInt,PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyGetSubcellIDGivenCellIdVertexIdFaceId(TDy,PetscInt,PetscInt,PetscInt,PetscInt*);

#endif
