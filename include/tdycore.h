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

PETSC_EXTERN PetscErrorCode TDyCreate(TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyCreateWithDM(DM dm,TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyView(TDy tdy,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode TDySetFromOptions(TDy tdy);

PETSC_EXTERN PetscErrorCode TDyGetDimension(TDy tdy,PetscInt *dim);
PETSC_EXTERN PetscErrorCode TDyGetDM(TDy tdy,DM *dm);
PETSC_EXTERN PetscErrorCode TDyGetCentroidArray(TDy tdy,PetscReal **X);

PETSC_EXTERN PetscErrorCode TDySetPorosityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetThermalConductivityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetResidualSaturationFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetForcingFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetEnergyForcingFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetDirichletValueFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetTemperatureDirichletValueFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetDirichletFluxFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);

PETSC_EXTERN PetscErrorCode TDySetGravityVector(TDy,PetscReal*);

PETSC_EXTERN PetscErrorCode TDySetPermeabilityScalar  (TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityDiagonal(TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityTensor  (TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetCellPermeability(TDy,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode TDySetPorosity            (TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetSpecificHeatCapacity(TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetRockDensity         (TDy tdy,SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDySetPorosityValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetBlockPermeabilityValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetResidualSaturationValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetMaterialPropertyMValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetMaterialPropertyNValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetMaterialPropertyAlphaValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetSourceSinkValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDySetEnergySourceSinkValuesLocal(TDy,PetscInt,const PetscInt[],const PetscScalar[]);

PETSC_EXTERN PetscErrorCode TDyGetSaturationValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetLiquidMassValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetMaterialPropertyMValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetMaterialPropertyAlphaValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetPorosityValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetBlockPermeabilityValuesLocal(TDy,PetscInt*,PetscScalar[]);
PETSC_EXTERN PetscErrorCode TDyGetNumCellsLocal(TDy,PetscInt*);
PETSC_EXTERN PetscErrorCode TDyGetCellNaturalIDsLocal(TDy,PetscInt*,PetscInt[]);
PETSC_EXTERN PetscErrorCode TDyGetCellIsLocal(TDy,PetscInt*,PetscInt[]);


PETSC_EXTERN PetscErrorCode TDySetDirichletFlux    (TDy tdy,SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDyResetDiscretizationMethod(TDy tdy);

PETSC_EXTERN PetscErrorCode TDySetDiscretizationMethod(TDy tdy,
    TDyMethod method);
PETSC_EXTERN PetscErrorCode TDySetMode(TDy tdy, TDyMode mode);
PETSC_EXTERN PetscErrorCode TDySetup(TDy tdy);
PETSC_EXTERN PetscErrorCode TDySetQuadratureType(TDy tdy,
    TDyQuadratureType qtype);
PETSC_EXTERN PetscErrorCode TDySetWaterDensityType(TDy,TDyWaterDensityType);
PETSC_EXTERN PetscErrorCode TDySetMPFAOGmatrixMethod(TDy,TDyMPFAOGmatrixMethod);
PETSC_EXTERN PetscErrorCode TDySetMPFAOBoundaryConditionType(TDy,TDyMPFAOBoundaryConditionType);

PETSC_EXTERN PetscErrorCode TDyComputeSystem(TDy tdy,Mat K,Vec F);
PETSC_EXTERN PetscErrorCode TDySetIFunction(TS ts,TDy tdy);
PETSC_EXTERN PetscErrorCode TDySetIJacobian(TS ts,TDy tdy);
PETSC_EXTERN PetscErrorCode TDySetSNESFunction(SNES snes,TDy tdy);
PETSC_EXTERN PetscErrorCode TDySetSNESJacobian(SNES snes,TDy tdy);
PETSC_EXTERN PetscErrorCode TDyComputeErrorNorms(TDy tdy,Vec U,PetscReal *normp,
    PetscReal *normv);

PETSC_EXTERN PetscErrorCode TDySetDtimeForSNESSolver(TDy,PetscReal);
PETSC_EXTERN PetscErrorCode TDySetInitialSolutionForSNESSolver(TDy,Vec);
PETSC_EXTERN PetscErrorCode TDyPreSolveSNESSolver(TDy);
PETSC_EXTERN PetscErrorCode TDyPostSolveSNESSolver(TDy,Vec);

PETSC_EXTERN PetscErrorCode TDyOutputRegression(TDy,Vec);

PETSC_EXTERN PetscErrorCode TDyTPFInitialize(TDy tdy);
PETSC_EXTERN PetscErrorCode TDyTPFComputeSystem(TDy tdy,Mat K,Vec F);
PETSC_EXTERN PetscReal TDyTPFPressureNorm(TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyTPFVelocityNorm(TDy tdy,Vec U);
PETSC_EXTERN PetscErrorCode TDyTPFCheckMeshSuitability(TDy tdy);

PETSC_EXTERN PetscErrorCode TDyWYInitialize(TDy tdy);
PETSC_EXTERN PetscErrorCode TDyWYComputeSystem(TDy tdy,Mat K,Vec F);

PETSC_EXTERN PetscErrorCode TDyBDMInitialize(TDy tdy);
PETSC_EXTERN PetscErrorCode TDyBDMComputeSystem(TDy tdy,Mat K,Vec F);
PETSC_EXTERN PetscReal TDyBDMPressureNorm(TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyBDMVelocityNorm(TDy tdy,Vec U);

PETSC_EXTERN PetscErrorCode TDyWYRecoverVelocity(TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyWYPressureNorm(TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyWYVelocityNorm(TDy tdy);
PETSC_EXTERN PetscErrorCode TDyWYResidual(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,
    void *ctx);

PETSC_EXTERN void RelativePermeability_Mualem(PetscReal m,PetscReal Se,
    PetscReal *Kr,PetscReal *dKr_dSe);
PETSC_EXTERN void RelativePermeability_Irmay(PetscReal m,PetscReal Se,
    PetscReal *Kr,PetscReal *dKr_dSe);
PETSC_EXTERN PetscErrorCode TDyUpdateState(TDy tdy,PetscReal *P);

PETSC_EXTERN PetscErrorCode Pullback(PetscScalar *K,PetscScalar *DFinv,
                                     PetscScalar *Kappa,PetscScalar J,PetscInt nn);
PETSC_EXTERN PetscInt TDyGetNumberOfCellVertices(DM dm);
PETSC_EXTERN PetscInt TDyGetNumberOfFaceVertices(DM dm);
PETSC_EXTERN PetscReal TDyL1norm(PetscReal *x,PetscReal *y,PetscInt dim);
PETSC_EXTERN PetscReal TDyADotBMinusC(PetscReal *a,PetscReal *b,PetscReal *c,
                                      PetscInt dim);
PETSC_EXTERN PetscReal TDyADotB(PetscReal *a,PetscReal *b,PetscInt dim);
PETSC_EXTERN PetscErrorCode TDyCreateCellVertexMap(TDy tdy,PetscInt **_map);
PETSC_EXTERN PetscErrorCode TDyCreateCellVertexDirFaceMap(TDy tdy,
    PetscInt **_map);
PETSC_EXTERN PetscErrorCode TDyQuadrature(PetscQuadrature q,PetscInt dim);

PETSC_EXTERN void HdivBasisQuad(const PetscReal *x,PetscReal *B,PetscReal *DF,PetscReal J);
PETSC_EXTERN void HdivBasisHex(const PetscReal *x,PetscReal *B,PetscReal *DF,PetscReal J);
PETSC_EXTERN PetscErrorCode IntegrateOnFace(TDy tdy,PetscInt c,PetscInt f,
    PetscReal *integral);

PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem(TDy, Mat, Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity(TDy, Vec);
PETSC_EXTERN PetscReal TDyMPFAOVelocityNorm(TDy);
PETSC_EXTERN PetscReal TDyMPFAOPressureNorm(TDy tdy,Vec U);

/* ---------------------------------------------------------------- */

PETSC_EXTERN void PrintMatrix(PetscReal *A,PetscInt nr,PetscInt nc,
                              PetscBool row_major);
PETSC_EXTERN PetscErrorCode CheckSymmetric(PetscReal *A,PetscInt n);

PETSC_EXTERN PetscErrorCode TDyRichardsInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyPostSolveSNESSolver(TDy,Vec);

#endif
