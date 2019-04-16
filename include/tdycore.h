#if !defined(TDYCORE_H)
#define TDYCORE_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include "tdycoremesh.h"

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

typedef struct _TDyOps *TDyOps;
struct _TDyOps {
  PetscErrorCode (*create)(TDy);
  PetscErrorCode (*destroy)(TDy);
  PetscErrorCode (*view)(TDy);
  PetscErrorCode (*setup)(TDy);
  PetscErrorCode (*setfromoptions)(TDy);
};

struct _p_TDy {
  PETSCHEADER(struct _TDyOps);
  PetscBool setup;
  DM dm;

  /* arrays of the size of the Hasse diagram */
  PetscReal *V; /* volume of point (if applicable) */
  PetscReal *X; /* centroid of point */
  PetscReal *N; /* normal of point (if applicable) */

  /* problem constants */
  PetscReal  rho;        /* density of water [kg m-3]*/
  PetscReal  mu;         /* viscosity of water [Pa s] */
  PetscReal  Sr;         /* residual saturation (min) [1] */
  PetscReal  Ss;         /* saturated saturation (max) [1] */
  PetscReal  gravity[3]; /* vector of gravity [m s-2] */
  PetscReal  Pref;       /* reference pressure */


  /* material parameters */
  PetscReal *K,
            *K0;     /* permeability tensor (cell,intrinsic) for each cell [m2] */
  PetscReal *Kr;        /* relative permeability for each cell [1] */
  PetscReal *porosity;  /* porosity for each cell [1] */
  PetscReal *S,
            *dS_dP;  /* saturation and derivative wrt pressure for each cell [1] */
  SpatialFunction forcing;
  SpatialFunction dirichlet;
  SpatialFunction flux;

  /* method-specific information*/
  TDyMethod method;
  TDyQuadratureType qtype;
  PetscBool allow_unsuitable_mesh;

  /* Wheeler-Yotov */
  PetscInt  *vmap;      /* [cell,local_vertex] --> global_vertex */
  PetscInt  *emap;      /* [cell,local_vertex,direction] --> global_face */
  PetscInt  *fmap;      /* [face,local_vertex] --> global_vertex */
  PetscReal *Alocal;    /* local element matrices (Ku,v) */
  PetscReal *Flocal;    /* local element vectors (f,w) */
  PetscQuadrature quad; /* vertex-based quadrature rule */
  PetscReal *vel;       /* [face,local_vertex] --> velocity normal to face at vertex */

  PetscInt  *LtoG;
  PetscInt  *orient;
  PetscInt  *faces;

  /* MPFA-O */
  TDy_mesh *mesh;
  PetscReal ****subc_Gmatrix; /* Gmatrix for subcells */
  PetscReal ***Trans;

};

PETSC_EXTERN PetscClassId TDY_CLASSID;

PETSC_EXTERN PetscLogEvent TDy_ComputeSystem;

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode TDyCreate(DM dm,TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyView(TDy tdy,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode TDySetFromOptions(TDy tdy);

PETSC_EXTERN PetscErrorCode TDySetPermeabilityScalar  (TDy tdy,
    SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityDiagonal(TDy tdy,
    SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityTensor  (TDy tdy,
    SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPorosity            (TDy tdy,
    SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDySetForcingFunction  (TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetDirichletFunction(TDy tdy,SpatialFunction f);
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

/* ---------------------------------------------------------------- */

PETSC_EXTERN void PrintMatrix(PetscReal *A,PetscInt nr,PetscInt nc,
                              PetscBool row_major);
PETSC_EXTERN PetscErrorCode CheckSymmetric(PetscReal *A,PetscInt n);

#endif
