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

/// A TDyScalarSpatialFunction computes the value of a scalar quantity f at n
/// points x. Use this type to explicitly indicate that f is a scalar.
/// Functions of this type accept an integer n for the number of points, an
/// array x of length dim*n containing the coordinates of the points, and an
/// array f of length n that stores the resulting n scalar values.
typedef void (*TDyScalarSpatialFunction)(PetscInt n, PetscReal *x, PetscReal *f);

/// A TDyVectorSpatialFunction computes the value of a vector quantity (or perhaps
/// a diagonal anisotropic tensor) f at n points x. Use this type to explicitly
/// indicate that f is a vector. Functions of this type accept an integer n for
/// the number of points, an array x of length dim*n containing the coordinates
/// of the points, and an array f of length dim*n that stores the resulting n
/// vector values.
typedef void (*TDyVectorSpatialFunction)(PetscInt n, PetscReal *x, PetscReal *f);

/// A TDyTensorSpatialFunction computes the value of a rank-2 tensor quantity f at
/// n points x. Use this type to explicitly indicate that f is a full tensor.
/// Functions of this type accept an integer n for the number of points, an
/// array x of length dim*n containing the coordinates of the points, and an
/// array f of length dim*dim*n that stores the resulting n tensor values.
typedef void (*TDyTensorSpatialFunction)(PetscInt n, PetscReal *x, PetscReal *f);

/// A TDySpatialFunction has the same type as the above functions, but can be
/// used to store any one of them.
typedef void (*TDySpatialFunction)(PetscInt n, PetscReal *x, PetscReal *f);

typedef struct _p_TDy *TDy;

typedef enum {
  WATER_DENSITY_CONSTANT=0,
  WATER_DENSITY_EXPONENTIAL=1
} TDyWaterDensityType;

PETSC_EXTERN const char *const TDyWaterDensityTypes[];

PETSC_EXTERN PetscClassId TDY_CLASSID;

PETSC_EXTERN PetscLogEvent TDy_ComputeSystem;

// Process initialization
PETSC_EXTERN PetscErrorCode TDyInit(int, char*[]);
PETSC_EXTERN PetscErrorCode TDyInitNoArguments(void);
PETSC_EXTERN PetscErrorCode TDyOnFinalize(void (*)(void));
PETSC_EXTERN PetscErrorCode TDyFinalize(void);

// Registry of named functions
PetscErrorCode TDyRegisterFunction(const char*, TDySpatialFunction);
PetscErrorCode TDyGetFunction(const char*, TDySpatialFunction*);

// Dycore creation and lifecycle
PETSC_EXTERN PetscErrorCode TDyCreate(MPI_Comm, TDy*);
PETSC_EXTERN PetscErrorCode TDySetMode(TDy,TDyMode);
PETSC_EXTERN PetscErrorCode TDySetDiscretization(TDy,TDyDiscretization);
PETSC_EXTERN PetscErrorCode TDySetDMConstructor(TDy,void*,PetscErrorCode(*)(MPI_Comm, void*, DM*));
PETSC_EXTERN PetscErrorCode TDySetFromOptions(TDy);
PETSC_EXTERN PetscErrorCode TDySetup(TDy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy*);
PETSC_EXTERN PetscErrorCode TDyView(TDy,PetscViewer);

PETSC_EXTERN PetscErrorCode TDyGetDimension(TDy,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyGetDiscretization(TDy,TDyDiscretization*);
PETSC_EXTERN PetscErrorCode TDyGetDM(TDy,DM*);
PETSC_EXTERN PetscErrorCode TDyGetBoundaryFaces(TDy,PetscInt*, const PetscInt**);
PETSC_EXTERN PetscErrorCode TDyRestoreBoundaryFaces(TDy,PetscInt*, const PetscInt**);

// Set material properties: via PETSc operations
PETSC_EXTERN PetscErrorCode TDySetWaterDensityType(TDy,TDyWaterDensityType);
PETSC_EXTERN PetscErrorCode TDySetConstantPorosity(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetPorosityFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantIsotropicPermeability(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetIsotropicPermeabilityFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantDiagonalPermeability(TDy,PetscReal[]);
PETSC_EXTERN PetscErrorCode TDySetDiagonalPermeabilityFunction(TDy,TDyVectorSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantTensorPermeability(TDy,PetscReal[]);
PETSC_EXTERN PetscErrorCode TDySetTensorPermeabilityFunction(TDy,TDyTensorSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantIsotropicThermalConductivity(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetIsotropicThermalConductivityFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantDiagonalThermalConductivity(TDy,PetscReal[]);
PETSC_EXTERN PetscErrorCode TDySetDiagonalThermalConductivityFunction(TDy,TDyVectorSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantTensorThermalConductivity(TDy,PetscReal[]);
PETSC_EXTERN PetscErrorCode TDySetTensorThermalConductivityFunction(TDy,TDyTensorSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantResidualSaturation(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetResidualSaturationFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantSoilDensity(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetSoilDensityFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetConstantSoilSpecificHeat(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetSoilSpecificHeatFunction(TDy,TDyScalarSpatialFunction);

// Set boundary conditions and sources/sinks
PETSC_EXTERN PetscErrorCode TDySetForcingFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetEnergyForcingFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetBoundaryPressureFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetBoundaryTemperatureFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySetBoundaryVelocityFunction(TDy,TDyScalarSpatialFunction);
PETSC_EXTERN PetscErrorCode TDySelectForcingFunction(TDy,const char*);
PETSC_EXTERN PetscErrorCode TDySelectEnergyForcingFunction(TDy,const char*);
PETSC_EXTERN PetscErrorCode TDySelectBoundaryPressureFunction(TDy,const char*);
PETSC_EXTERN PetscErrorCode TDySelectBoundaryTemperatureFunction(TDy,const char*);
PETSC_EXTERN PetscErrorCode TDySelectBoundaryVelocityFunction(TDy,const char*);

PETSC_EXTERN PetscErrorCode TDyUpdateState(TDy,PetscReal*,PetscInt);
PETSC_EXTERN PetscErrorCode TDyComputeErrorNorms(TDy,Vec,PetscReal*,PetscReal*);

// Access to diagnostic variables
PETSC_EXTERN PetscErrorCode TDyUpdateDiagnostics(TDy);
PETSC_EXTERN PetscErrorCode TDyCreateDiagnosticVector(TDy,Vec*);
PETSC_EXTERN PetscErrorCode TDyGetLiquidSaturation(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDyGetLiquidMass(TDy,Vec);

// We will remove the following functions in favor of setting function pointers
// that a given solver uses to extract info from a DM.
PETSC_EXTERN PetscErrorCode TDySetIFunction(TS,TDy);
PETSC_EXTERN PetscErrorCode TDySetIJacobian(TS,TDy);
PETSC_EXTERN PetscErrorCode TDySetSNESFunction(SNES,TDy);
PETSC_EXTERN PetscErrorCode TDySetSNESJacobian(SNES,TDy);

PETSC_EXTERN PetscErrorCode TDySetDtimeForSNESSolver(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetInitialCondition(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDySetPreviousSolutionForSNESSolver(TDy,Vec);
PETSC_INTERN PetscErrorCode TDyPreSolveSNESSolver(TDy);

PETSC_EXTERN PetscErrorCode TDyOutputRegression(TDy,Vec);

PETSC_EXTERN PetscInt TDyGetNumberOfCellVertices(DM);
PETSC_EXTERN PetscInt TDyGetNumberOfFaceVertices(DM);
PETSC_EXTERN PetscReal TDyL1norm(PetscReal*,PetscReal*,PetscInt);
PETSC_EXTERN PetscReal TDyADotBMinusC(PetscReal*,PetscReal*,PetscReal*,PetscInt);
PETSC_EXTERN PetscReal TDyADotB(PetscReal*,PetscReal*,PetscInt);

/* ---------------------------------------------------------------- */

PETSC_EXTERN void PrintMatrix(PetscReal*,PetscInt,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode CheckSymmetric(PetscReal*,PetscInt);

PETSC_EXTERN PetscErrorCode TDyPostSolve(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDyCreateVectors(TDy);
PETSC_EXTERN PetscErrorCode TDyCreateJacobian(TDy);

PETSC_EXTERN PetscErrorCode TDyTimeIntegratorRunToTime(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorSetTimeStep(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDyTimeIntegratorOutputRegression(TDy);

PETSC_EXTERN PetscErrorCode TDyDriverInitializeTDy(TDy);

//-------------------------------------------------
// Multi-point Flux Approximation (MPFA-O) methods
//-------------------------------------------------

PETSC_EXTERN PetscErrorCode TDyMPFAOSetGmatrixMethod(TDy,TDyMPFAOGmatrixMethod);
PETSC_EXTERN PetscErrorCode TDyMPFAOSetBoundaryConditionType(TDy,TDyMPFAOBoundaryConditionType);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem(TDy,Mat,Vec);

//------------------------
// Finite element methods
//------------------------

typedef enum {
  LUMPED=0,
  FULL
} TDyQuadratureType;

PETSC_EXTERN const char *const TDyQuadratureTypes[];

PETSC_EXTERN PetscErrorCode TDyWYSetQuadrature(TDy,TDyQuadratureType);
PETSC_EXTERN PetscErrorCode TDyWYComputeSystem(TDy,Mat,Vec);

PETSC_EXTERN PetscErrorCode TDyBDMSetQuadrature(TDy,TDyQuadratureType);
PETSC_EXTERN PetscErrorCode TDyBDMComputeSystem(TDy,Mat,Vec);

#endif
