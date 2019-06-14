#if !defined(TDYCORE_H)
#define TDYCORE_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

/* ---------------------------------------------------------------- */

typedef enum {
  TPF=0,                /* two point flux, classic finite volumes                  */
  MPFA_O,               /* multipoint flux approximation - O method                */
  BDM,                  /* P0,BDM1 spaces, standard approach                       */
  WY                    /* P0,BDM1 spaces, vertex quadrature, statically condensed */
} TDyMethod;

PETSC_EXTERN const char *const TDyMethods[];

typedef enum {
  LUMPED=0,
  FULL
} TDyQuadratureType;

PETSC_EXTERN const char *const TDyQuadratureTypes[];

typedef void (*SpatialFunction)(PetscReal *x,PetscReal *f); /* returns f(x) */

typedef struct _p_TDy *TDy;


PETSC_EXTERN PetscClassId TDY_CLASSID;

PETSC_EXTERN PetscLogEvent TDy_ComputeSystem;

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode TDyCreate(DM dm,TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyView(TDy tdy,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode TDySetFromOptions(TDy tdy);

PETSC_EXTERN PetscErrorCode TDyGetDimension(TDy tdy,PetscInt *dim);
PETSC_EXTERN PetscErrorCode TDyGetDM(TDy tdy,DM *dm);

PETSC_EXTERN PetscErrorCode TDySetPermeabilityFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetForcingFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetDirichletValueFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TDySetDirichletFluxFunction(TDy,PetscErrorCode(*)(TDy,PetscReal*,PetscReal*,void*),void*);

PETSC_EXTERN PetscErrorCode TDySetPermeabilityScalar  (TDy tdy,
    SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityDiagonal(TDy tdy,
    SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityTensor  (TDy tdy,
    SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPorosity            (TDy tdy,
    SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDySetDirichletFlux    (TDy tdy,SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDyResetDiscretizationMethod(TDy tdy);

PETSC_EXTERN PetscErrorCode TDySetDiscretizationMethod(TDy tdy,
    TDyMethod method);
PETSC_EXTERN PetscErrorCode TDySetQuadratureType(TDy tdy,
    TDyQuadratureType qtype);

PETSC_EXTERN PetscErrorCode TDyComputeSystem(TDy tdy,Mat K,Vec F);
PETSC_EXTERN PetscErrorCode TDySetIFunction(TS ts,TDy tdy);
PETSC_EXTERN PetscErrorCode TDyComputeErrorNorms(TDy tdy,Vec U,PetscReal *normp,
    PetscReal *normv);

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

PETSC_EXTERN void RelativePermeability_Irmay(PetscReal m,PetscReal Se,
    PetscReal *Kr,PetscReal *dKr_dSe);
PETSC_EXTERN void PressureSaturation_Gardner(PetscReal n,PetscReal m,
    PetscReal alpha,PetscReal Pc,PetscReal *Se,PetscReal *dSe_dPc);
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

PETSC_EXTERN void HdivBasisQuad(const PetscReal *x,PetscReal *B);
PETSC_EXTERN void HdivBasisHex(const PetscReal *x,PetscReal *B);
PETSC_EXTERN PetscErrorCode IntegrateOnFace(TDy tdy,PetscInt c,PetscInt f,
    PetscReal *integral);

PETSC_EXTERN PetscErrorCode TDyMPFAOInitialize(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem(TDy, Mat, Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity(TDy, Vec);
PETSC_EXTERN PetscReal TDyMPFAOVelocityNorm(TDy);
PETSC_EXTERN PetscReal TDyMPFAOPressureNorm(TDy tdy,Vec U);

/* ---------------------------------------------------------------- */

PETSC_EXTERN void PrintMatrix(PetscReal *A,PetscInt nr,PetscInt nc,
                              PetscBool row_major);
PETSC_EXTERN PetscErrorCode CheckSymmetric(PetscReal *A,PetscInt n);

#endif
