#if !defined(TDYCORE_H)
#define TDYCORE_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>

/* ---------------------------------------------------------------- */

typedef struct _p_TDy *TDy;

typedef enum {
  TWO_POINT_FLUX=0,     /* classic finite volumes                                  */
  MULTIPOINT_FLUX,      /*                                                         */
  MIXED_FINITE_ELEMENT, /* P0,BDM1 spaces, standard approach                       */
  WHEELER_YOTOV         /* P0,BDM1 spaces, vertex quadrature, statically condensed */
} TDyMethod;

PETSC_EXTERN const char *const TDyMethods[];

typedef void (*SpatialFunction)(PetscReal *x,PetscReal *f); /* returns f(x) */

struct _p_TDy {
  
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
  PetscReal *K,*K0;     /* permeability tensor (cell,intrinsic) for each cell [m2] */
  PetscReal *Kr;        /* relative permeability for each cell [1] */
  PetscReal *porosity;  /* porosity for each cell [1] */
  PetscReal *S,*dS_dP;  /* saturation and derivative wrt pressure for each cell [1] */
  SpatialFunction forcing;
  SpatialFunction dirichlet;
  SpatialFunction flux;
  
  /* method-specific information*/
  TDyMethod method;
  
  /* Wheeler-Yotov */
  PetscInt  *vmap;      /* [cell,local_vertex] --> global_vertex */
  PetscInt  *emap;      /* [cell,local_vertex,direction] --> global_face */
  PetscInt  *fmap;      /* [face,local_vertex] --> global_vertex */
  PetscReal *Alocal;    /* local element matrices (Ku,v) */
  PetscReal *Flocal;    /* local element vectors (f,w) */
  PetscQuadrature quad; /* vertex-based quadrature rule */
  PetscReal *vel;       /* [face,local_vertex] --> velocity normal to face at vertex */
};

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode TDyCreate(DM dm,TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy *tdy);

PETSC_EXTERN PetscErrorCode TDySetPermeabilityScalar  (DM dm,TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityDiagonal(DM dm,TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityTensor  (DM dm,TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPorosity            (DM dm,TDy tdy,SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDySetForcingFunction  (TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetDirichletFunction(TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetDirichletFlux    (TDy tdy,SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDyResetDiscretizationMethod(TDy tdy);
PETSC_EXTERN PetscErrorCode TDySetDiscretizationMethod(DM dm,TDy tdy,TDyMethod method);

PETSC_EXTERN PetscErrorCode TDyComputeSystem(DM dm,TDy tdy,Mat K,Vec F);
PETSC_EXTERN PetscErrorCode TDySetIFunction(TS ts,TDy tdy);

PETSC_EXTERN PetscErrorCode TDyWYInitialize(DM dm,TDy tdy);
PETSC_EXTERN PetscErrorCode TDyWYComputeSystem(DM dm,TDy tdy,Mat K,Vec F);

PETSC_EXTERN PetscErrorCode TDyWYRecoverVelocity(DM dm,TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyWYPressureNorm(DM dm,TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyWYVelocityNorm(DM dm,TDy tdy);
PETSC_EXTERN PetscReal TDyWYDivergenceNorm(DM dm,TDy tdy);
PETSC_EXTERN PetscErrorCode TDyWYResidual(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx);

PETSC_EXTERN void RelativePermeability_Irmay(PetscReal m,PetscReal Se,PetscReal *Kr,PetscReal *dKr_dSe);
PETSC_EXTERN void PressureSaturation_Gardner(PetscReal n,PetscReal m,PetscReal alpha,PetscReal Pc,PetscReal *Se,PetscReal *dSe_dPc);
PETSC_EXTERN PetscErrorCode TDyUpdateState(DM dm,TDy tdy,PetscReal *P);

/* ---------------------------------------------------------------- */

PETSC_EXTERN void PrintMatrix(PetscReal *A,PetscInt nr,PetscInt nc,PetscBool row_major);
PETSC_EXTERN PetscErrorCode CheckSymmetric(PetscReal *A,PetscInt n);

#endif
