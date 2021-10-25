#if !defined(TDYCORE_H)
#define TDYCORE_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

/* ---------------------------------------------------------------- */

/// This type lists modes that identify the set of governing equations solved
/// by the dycore.
typedef enum {
  /// Richards equation
  RICHARDS=0,
  /// Non-isothermal flows
  TH
} TDyMode;

PETSC_EXTERN const char *const TDyModes[];

/// This type enumerates discretizations supported by the dycore.
typedef enum {
  /// multi-point flux approximation - O method
  MPFA_O=0,
  /// multi-point flux approximation - O method using DAE
  MPFA_O_DAE,
  /// multipoint flux approximation - O method using TS transient (conservative)
  /// approach
  MPFA_O_TRANSIENTVAR,
  /// finite element using P0, BDM1 spaces, standard approach
  BDM,
  /// finite element using P0,BDM1 spaces, vertex quadrature, statically
  /// condensed
  WY
} TDyDiscretization;

PETSC_EXTERN const char *const TDyDiscretizations[];

typedef enum {
  MPFAO_GMATRIX_DEFAULT=0, /* default method to compute gmatrix for MPFA-O method        */
  MPFAO_GMATRIX_TPF        /* two-point flux method to compute gmatrix for MPFA-O method */
} TDyMPFAOGmatrixMethod;

PETSC_EXTERN const char *const TDyMPFAOGmatrixMethods[];

typedef enum {
  MPFAO_DIRICHLET_BC=0,  /* Dirichlet boundary condiiton */
  MPFAO_NEUMANN_BC,       /* Neumann zero-flux boundary condition */
  MPFAO_SEEPAGE_BC       /* Seepage boundary condition */
} TDyMPFAOBoundaryConditionType;

PETSC_EXTERN const char *const TDyMPFAOBoundaryConditionTypes[];

typedef enum {
  LUMPED=0,
  FULL
} TDyQuadratureType;

PETSC_EXTERN const char *const TDyQuadratureTypes[];

typedef enum {
  TDySNES=0,
  TDyTS
} TDyTimeIntegrationMethod;

typedef enum {
  TDyCreated=0x0,
  TDyParametersInitialized=0x1,
  TDyModeSet=0x1<<1,
  TDyDiscretizationSet=0x1<<2,
  TDyOptionsSet=0x1<<3,
  TDySetupFinished=0x1<<4,
} TDySetupFlags;

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
PETSC_EXTERN PetscErrorCode TDyInit(int, char*[]);
PETSC_EXTERN PetscErrorCode TDyInitNoArguments(void);
PETSC_EXTERN PetscErrorCode TDyOnFinalize(void (*)(void));
PETSC_EXTERN PetscErrorCode TDyFinalize(void);

PETSC_EXTERN PetscErrorCode TDyCreate(MPI_Comm, TDy*);
PETSC_EXTERN PetscErrorCode TDySetMode(TDy,TDyMode);
PETSC_EXTERN PetscErrorCode TDySetDiscretization(TDy,TDyDiscretization);
PETSC_EXTERN PetscErrorCode TDySetDMConstructor(TDy,void*,PetscErrorCode(*)(MPI_Comm, void*, DM*));
PETSC_EXTERN PetscErrorCode TDySetFromOptions(TDy);
PETSC_EXTERN PetscErrorCode TDySetup(TDy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy*);
PETSC_EXTERN PetscErrorCode TDyView(TDy,PetscViewer);

PETSC_EXTERN PetscErrorCode TDyGetDimension(TDy,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyGetDM(TDy,DM*);
PETSC_EXTERN PetscErrorCode TDyGetBoundaryFaces(TDy,PetscInt*, const PetscInt**);
PETSC_EXTERN PetscErrorCode TDyRestoreBoundaryFaces(TDy,PetscInt*, const PetscInt**);
PETSC_EXTERN PetscErrorCode TDyGetCentroidArray(TDy,PetscReal**);

PETSC_EXTERN PetscErrorCode TDySetGravityVector(TDy,PetscReal*);

// Set material properties: via PETSc operations
PETSC_EXTERN PetscErrorCode TDySetPorosityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetThermalConductivityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetResidualSaturationFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetSoilDensityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetSoilSpecificHeatFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);

// Set boundary and source-sink: via PETSc operations
PetscErrorCode TDyRegisterFunction(const char*, PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*));
PETSC_EXTERN PetscErrorCode TDySetForcingFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetEnergyForcingFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySelectBoundaryPressureFn(TDy,const char*,void*);
PETSC_EXTERN PetscErrorCode TDySelectBoundaryTemperatureFn(TDy,const char*,void*);
PETSC_EXTERN PetscErrorCode TDySelectBoundaryVelocityFn(TDy,const char*,void*);
PETSC_EXTERN PetscErrorCode TDySetBoundaryPressureFn(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetBoundaryTemperatureFn(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetBoundaryVelocityFn(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);

// Set material properties: via spatial function
PETSC_EXTERN PetscErrorCode TDySetPermeabilityScalar  (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityDiagonal(TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityTensor  (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetCellPermeability(TDy,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode TDySetPorosity            (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetSoilSpecificHeat    (TDy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetSoilDensity         (TDy,SpatialFunction f);

// Set material properties: For each cell
PETSC_EXTERN PetscErrorCode TDySetPorosityValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetBlockPermeabilityValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetResidualSaturationValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetCharacteristicCurveMualemValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetCharacteristicCurveNValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetCharacteristicCurveVanGenuchtenValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[],const PetscScalar[]);

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

PETSC_EXTERN PetscErrorCode TDyResetDiscretization(TDy);

PETSC_EXTERN PetscErrorCode TDySetQuadratureType(TDy,TDyQuadratureType);
PETSC_EXTERN PetscErrorCode TDySetWaterDensityType(TDy,TDyWaterDensityType);
PETSC_EXTERN PetscErrorCode TDySetMPFAOGmatrixMethod(TDy,TDyMPFAOGmatrixMethod);
PETSC_EXTERN PetscErrorCode TDySetMPFAOBoundaryConditionType(TDy,TDyMPFAOBoundaryConditionType);

// We will probably remove the following functions.
PETSC_EXTERN PetscErrorCode TDyComputeSystem(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDySetIFunction(TS,TDy);
PETSC_EXTERN PetscErrorCode TDySetIJacobian(TS,TDy);
PETSC_EXTERN PetscErrorCode TDySetSNESFunction(SNES,TDy);
PETSC_EXTERN PetscErrorCode TDySetSNESJacobian(SNES,TDy);

PETSC_EXTERN PetscErrorCode TDyComputeErrorNorms(TDy,Vec,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode TDySetDtimeForSNESSolver(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetInitialCondition(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDySetPreviousSolutionForSNESSolver(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDyPreSolveSNESSolver(TDy);
PETSC_EXTERN PetscErrorCode TDyPostSolveSNESSolver(TDy,Vec);

PETSC_EXTERN PetscErrorCode TDyOutputRegression(TDy,Vec);

PETSC_EXTERN PetscErrorCode TDyWYComputeSystem(TDy,Mat,Vec);

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
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorSetTimeStep(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorOutputRegression(TDy);

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

PETSC_INTERN PetscErrorCode TDyMeshGetCellVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellEdges(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNeighbors(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellCentroid(TDyMesh*, PetscInt, TDyCoordinate*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellVolume(TDyMesh*, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumVertices(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumVertices(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumFaces(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetCellNumNeighbors(TDyMesh*, PetscInt, PetscInt*);

PETSC_INTERN PetscErrorCode TDyMeshGetVertexInternalCells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexSubcells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexSubfaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexBoundaryFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumInternalCells(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumSubcells(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumFaces(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetVertexNumBoundaryFaces(TDyMesh*, PetscInt, PetscInt*);

PETSC_INTERN PetscErrorCode TDyMeshGetFaceCells(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceCentroid(TDyMesh*, PetscInt, TDyCoordinate*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceNormal(TDyMesh*, PetscInt, TDyVector*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceArea(TDyMesh*, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceNumCells(TDyMesh*, PetscInt, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetFaceNumVertices(TDyMesh*, PetscInt, PetscInt*);

PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaces(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellIsFaceUp(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceUnknownIdxs(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceFluxIdxs(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceAreas(TDyMesh*, PetscInt, PetscReal**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellVertices(TDyMesh*, PetscInt, PetscInt**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellNuVectors(TDyMesh*, PetscInt, TDyVector**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellNuStarVectors(TDyMesh*, PetscInt, TDyVector**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellVariableContinutiyCoordinates(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellFaceCentroids(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellVerticesCoordinates(TDyMesh*, PetscInt, TDyCoordinate**, PetscInt*);
PETSC_INTERN PetscErrorCode TDyMeshGetSubcellNumFaces(TDyMesh*, PetscInt, PetscInt*);

#endif
