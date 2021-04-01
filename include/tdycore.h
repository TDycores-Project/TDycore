#if !defined(TDYCORE_H)
#define TDYCORE_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

/* ---------------------------------------------------------------- */

typedef enum {
  TPF=0,                /* two point flux, classic finite volumes                                              */
  MPFA_O,               /* multipoint flux approximation - O method                                            */
  MPFA_O_DAE,           /* multipoint flux approximation - O method using DAE                                  */
  MPFA_O_TRANSIENTVAR,  /* multipoint flux approximation - O method using TS transient (conservative) approach */
  BDM,                  /* P0,BDM1 spaces, standard approach                                                   */
  WY                    /* P0,BDM1 spaces, vertex quadrature, statically condensed                             */
} TDyMethod;

PETSC_EXTERN const char *const TDyMethods[];

typedef enum {
  MPFAO_GMATRIX_DEFAULT=0, /* default method to compute gmatrix for MPFA-O method        */
  MPFAO_GMATRIX_TPF        /* two-point flux method to compute gmatrix for MPFA-O method */
} TDyMPFAOGmatrixMethod;

PETSC_EXTERN const char *const TDyMPFAOGmatrixMethods[];

typedef enum {
  MPFAO_DIRICHLET_BC=0,  /* Dirichlet boundary condiiton */
  MPFAO_NEUMANN_BC       /* Neumann zero-flux boundary condition */
} TDyMPFAOBoundaryConditionType;

PETSC_EXTERN const char *const TDyMPFAOBoundaryConditionTypes[];

typedef enum {
  LUMPED=0,
  FULL
} TDyQuadratureType;

PETSC_EXTERN const char *const TDyQuadratureTypes[];

typedef enum {
  RICHARDS=0,
  TH
} TDyMode;

typedef enum {
  TDySNES=0,
  TDyTS
} TDyTimeIntegrationMethod;

typedef enum {
  TDyCreated=0x0,
  TDyParametersInitialized=0x1,
  TDyOptionsSet=0x2,
  TDySetupFinished=0x4,
} TDySetupFlags;

PETSC_EXTERN const char *const TDyModes[];

typedef void (*SpatialFunction)(PetscReal *x,PetscReal *f); /* returns f(x) */

typedef struct _p_TDy *TDy;

typedef enum {
  WATER_DENSITY_CONSTANT=0,
  WATER_DENSITY_EXPONENTIAL=1
} TDyWaterDensityType;

PETSC_EXTERN const char *const TDyWaterDensityTypes[];

PETSC_EXTERN PetscClassId TDY_CLASSID;

PETSC_EXTERN PetscLogEvent TDy_ComputeSystem;

/* ---------------------------------------------------------------- */
PETSC_EXTERN PetscErrorCode TDyInit(int argc, char* argv[]);
PETSC_EXTERN PetscErrorCode TDyInitNoArguments(void);
PETSC_EXTERN PetscErrorCode TDyFinalize(void);

PETSC_EXTERN PetscErrorCode TDyCreate(TDy*);
PETSC_EXTERN PetscErrorCode TDySetMode(TDy,TDyMode);
PETSC_EXTERN PetscErrorCode TDySetDiscretizationMethod(TDy,TDyMethod);
PETSC_EXTERN PetscErrorCode TDySetDM(TDy,DM);
PETSC_EXTERN PetscErrorCode TDySetFromOptions(TDy);
PETSC_EXTERN PetscErrorCode TDySetupNumericalMethods(TDy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyView(TDy,PetscViewer viewer);

PETSC_EXTERN PetscErrorCode TDyGetDimension(TDy,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyGetDM(TDy,DM*);
PETSC_EXTERN PetscErrorCode TDyGetCentroidArray(TDy,PetscReal**);

PETSC_EXTERN PetscErrorCode TDySetGravityVector(TDy,PetscReal*);

// Set material properties: via PETSc operations
PETSC_EXTERN PetscErrorCode TDySetPorosityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetThermalConductivityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetResidualSaturationFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);

// Set boundary and source-sink: via PETSc operations
PETSC_EXTERN PetscErrorCode TDySetForcingFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetEnergyForcingFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetDirichletValueFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetTemperatureDirichletValueFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetDirichletFluxFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);

// Set material properties: via spatial function
PETSC_EXTERN PetscErrorCode TDySetPermeabilityScalar  (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityDiagonal(TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityTensor  (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetCellPermeability(TDy,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode TDySetPorosity            (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetSoilSpecificHeat    (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetSoilDensity         (TDy,SpatialFunction f);

// Set boundary condition: via spatial function
PETSC_EXTERN PetscErrorCode TDySetDirichletFlux(TDy,SpatialFunction);

// Set material properties: For each cell
PETSC_EXTERN PetscErrorCode TDySetPorosityValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetBlockPermeabilityValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetResidualSaturationValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetCharacteristicCurveMValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetCharacteristicCurveNValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetCharacteristicCurveAlphaValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);

// Set condition: for each cell
PETSC_EXTERN PetscErrorCode TDySetSourceSinkValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetEnergySourceSinkValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);

// Get material value: for each cell
PETSC_EXTERN PetscErrorCode TDyGetSaturationValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetLiquidMassValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetCharacteristicCurveMValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetCharacteristicCurveAlphaValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetPorosityValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetBlockPermeabilityValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetNumCellsLocal(TDy,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyGetCellNaturalIDsLocal(TDy,PetscInt*,PetscInt[]);
PETSC_EXTERN PetscErrorCode TDyGetCellIsLocal(TDy,PetscInt*,PetscInt[]);


PETSC_EXTERN PetscErrorCode TDyResetDiscretizationMethod(TDy);

PETSC_EXTERN PetscErrorCode TDySetQuadratureType(TDy,TDyQuadratureType);
PETSC_EXTERN PetscErrorCode TDySetWaterDensityType(TDy,TDyWaterDensityType);
PETSC_EXTERN PetscErrorCode TDySetMPFAOGmatrixMethod(TDy,TDyMPFAOGmatrixMethod);
PETSC_EXTERN PetscErrorCode TDySetMPFAOBoundaryConditionType(TDy,TDyMPFAOBoundaryConditionType);

PETSC_EXTERN PetscErrorCode TDyComputeSystem(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDySetIFunction(TS,TDy);
PETSC_EXTERN PetscErrorCode TDySetIJacobian(TS,TDy);
PETSC_EXTERN PetscErrorCode TDySetSNESFunction(SNES,TDy);
PETSC_EXTERN PetscErrorCode TDySetSNESJacobian(SNES,TDy);
PETSC_EXTERN PetscErrorCode TDyComputeErrorNorms(TDy,Vec,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode TDySetDtimeForSNESSolver(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetPreviousSolutionForSNESSolver(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDyPreSolveSNESSolver(TDy);
PETSC_EXTERN PetscErrorCode TDyPostSolveSNESSolver(TDy,Vec);

PETSC_EXTERN PetscErrorCode TDyOutputRegression(TDy,Vec);

PETSC_EXTERN PetscErrorCode TDyTPFInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyTPFComputeSystem(TDy,Mat,Vec);
PETSC_EXTERN PetscReal TDyTPFPressureNorm(TDy,Vec);
PETSC_EXTERN PetscReal TDyTPFVelocityNorm(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDyTPFCheckMeshSuitability(TDy);

PETSC_EXTERN PetscErrorCode TDyWYInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyWYComputeSystem(TDy,Mat,Vec);

PETSC_EXTERN PetscErrorCode TDyBDMInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyBDMComputeSystem(TDy,Mat,Vec);
PETSC_EXTERN PetscReal TDyBDMPressureNorm(TDy,Vec);
PETSC_EXTERN PetscReal TDyBDMVelocityNorm(TDy,Vec);

PETSC_EXTERN PetscErrorCode TDyWYRecoverVelocity(TDy,Vec);
PETSC_EXTERN PetscReal TDyWYPressureNorm(TDy,Vec);
PETSC_EXTERN PetscReal TDyWYVelocityNorm(TDy);
PETSC_EXTERN PetscErrorCode TDyWYResidual(TS,PetscReal,Vec,Vec,Vec,void *ctx);

PETSC_EXTERN PetscErrorCode TDyUpdateState(TDy,PetscReal*);

PETSC_EXTERN PetscErrorCode Pullback(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar,PetscInt);
PETSC_EXTERN PetscInt TDyGetNumberOfCellVertices(DM);
PETSC_EXTERN PetscInt TDyGetNumberOfFaceVertices(DM);
PETSC_EXTERN PetscReal TDyL1norm(PetscReal*,PetscReal*,PetscInt);
PETSC_EXTERN PetscReal TDyADotBMinusC(PetscReal*,PetscReal*,PetscReal*,PetscInt);
PETSC_EXTERN PetscReal TDyADotB(PetscReal*,PetscReal*,PetscInt);
PETSC_EXTERN PetscErrorCode TDyCreateCellVertexMap(TDy,PetscInt**);
PETSC_EXTERN PetscErrorCode TDyCreateCellVertexDirFaceMap(TDy,PetscInt**);
PETSC_EXTERN PetscErrorCode TDyQuadrature(PetscQuadrature,PetscInt);

PETSC_EXTERN void HdivBasisQuad(const PetscReal*,PetscReal*,PetscReal*,PetscReal);
PETSC_EXTERN void HdivBasisHex(const PetscReal*,PetscReal*,PetscReal*,PetscReal);
PETSC_EXTERN PetscErrorCode IntegrateOnFace(TDy,PetscInt,PetscInt,PetscReal*);

PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity(TDy,Vec);
PETSC_EXTERN PetscReal TDyMPFAOVelocityNorm(TDy);
PETSC_EXTERN PetscReal TDyMPFAOPressureNorm(TDy,Vec);

/* ---------------------------------------------------------------- */

PETSC_EXTERN void PrintMatrix(PetscReal*,PetscInt,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode CheckSymmetric(PetscReal*,PetscInt);

PETSC_EXTERN PetscErrorCode TDyPostSolveSNESSolver(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDyCreateVectors(TDy);
PETSC_EXTERN PetscErrorCode TDyCreateJacobian(TDy);

PETSC_EXTERN PetscErrorCode TDyTimeIntegratorRunToTime(TDy,PetscReal);

PETSC_EXTERN PetscErrorCode TDyDriverInitializeTDy(TDy);

/* ---------------------------------------------------------------- */

typedef struct TDyMesh TDyMesh;

typedef struct {
  PetscReal X[3];
} TDyCoordinate;

typedef struct {
  PetscReal V[3];
} TDyVector;

PETSC_EXTERN PetscErrorCode TDyMeshCreateFromDM(DM, TDyMesh*);
PETSC_EXTERN PetscErrorCode TDyMeshDestroy(TDyMesh*);

PETSC_EXTERN PetscErrorCode TDyMeshGetCellVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetCellEdges(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetCellFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetCellNeighbors(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetCellCentroid(TDyMesh*, PetscInt, TDyCoordinate*);
PETSC_EXTERN PetscErrorCode TDyMeshGetCellVolume(TDyMesh*, PetscInt, PetscReal*);

PETSC_EXTERN PetscErrorCode TDyMeshGetVertexInternalCells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetVertexSubcells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetVertexFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetVertexSubfaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetVertexBoundaryFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);

PETSC_EXTERN PetscErrorCode TDyMeshGetFaceCells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetFaceVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetFaceCentroid(TDyMesh*, PetscInt, TDyCoordinate*);
PETSC_EXTERN PetscErrorCode TDyMeshGetFaceNormal(TDyMesh*, PetscInt, TDyVector*);
PETSC_EXTERN PetscErrorCode TDyMeshGetFaceArea(TDyMesh*, PetscInt, PetscReal*);

PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellIsFaceUp(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellFaceUnknownIdxs(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellFaceFluxIdxs(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellFaceAreas(TDyMesh*, PetscInt, PetscReal**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellNuVectors(TDyMesh*, PetscInt, TDyVector**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellNuStarVectors(TDyMesh*, PetscInt, TDyVector**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellVariableContinutiyCoordinates(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellFaceCentroids(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_EXTERN PetscErrorCode TDyMeshGetSubcellVerticesCoordinates(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);

#endif
