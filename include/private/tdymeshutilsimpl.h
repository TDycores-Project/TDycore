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
PETSC_INTERN PetscErrorCode AreFacesNeighbors(TDy_face*, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthNuVector(TDySubcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthFaceCentroid(TDySubcell*, PetscInt, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDySubCell_GetFaceIndexForAFace(TDySubcell*, PetscInt, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyEdge_GetCentroid(TDy_edge*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyFace_GetCentroid(TDy_face*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyFace_GetNormal(TDy_face*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyVertex_GetCoordinate(TDy_vertex*, PetscInt, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode TDyCell_GetCentroid2(TDyCell*, PetscInt, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode FindNeighboringVerticesOfAFace(TDy_face*, PetscInt, PetscInt, PetscInt[2]);
PETSC_INTERN PetscErrorCode SetupCell2CellConnectivity(TDy_vertex*, PetscInt, TDyCell*, TDy_face*, TDySubcell*, PetscInt**);
PETSC_INTERN PetscErrorCode FindFaceIDsOfACellCommonToAVertex(PetscInt, TDy_face*, TDy_vertex*, PetscInt,PetscInt[3],PetscInt*);
PETSC_INTERN PetscErrorCode IdentifyLocalCells(TDy);
PETSC_INTERN PetscErrorCode IdentifyLocalVertices(TDy);
PETSC_INTERN PetscErrorCode IdentifyLocalEdges(TDy);
PETSC_INTERN PetscErrorCode IdentifyLocalFaces(TDy);
PETSC_INTERN PetscErrorCode TDyFindSubcellOfACellThatIncludesAVertex(TDyCell*, PetscInt, TDy_vertex*, PetscInt, TDySubcell*, PetscInt*);
PETSC_INTERN PetscErrorCode TDyPrintSubcellInfo(TDy, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode TDyPrintFaceInfo(TDy, PetscInt);
PETSC_INTERN PetscErrorCode TDySubCell_GetIthNuStarVector(TDySubcell*,PetscInt,PetscInt,PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyEdge_GetNormal(TDy_edge*,PetscInt,PetscInt, PetscReal*);

#endif
