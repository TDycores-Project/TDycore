#if !defined(TDYCORE_H)
#define TDYCORE_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include "tdycoremesh.h"

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

  /* material parameters */
  PetscReal *K; /* allocate full tensor for each cell */
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

  /* MPFA-O */
  TDy_mesh mesh;

};

/* ---------------------------------------------------------------- */

PETSC_EXTERN PetscErrorCode TDyCreate(DM dm,TDy *tdy);
PETSC_EXTERN PetscErrorCode TDyDestroy(TDy *tdy);

PETSC_EXTERN PetscErrorCode TDySetPermeabilityScalar  (DM dm,TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityDiagonal(DM dm,TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetPermeabilityTensor  (DM dm,TDy tdy,SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDySetForcingFunction  (TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetDirichletFunction(TDy tdy,SpatialFunction f);
PETSC_EXTERN PetscErrorCode TDySetDirichletFlux    (TDy tdy,SpatialFunction f);

PETSC_EXTERN PetscErrorCode TDyResetDiscretizationMethod(TDy tdy);
PETSC_EXTERN PetscErrorCode TDySetDiscretizationMethod(DM dm,TDy tdy,TDyMethod method);

PETSC_EXTERN PetscErrorCode TDyComputeSystem(DM dm,TDy tdy,Mat K,Vec F);

PETSC_EXTERN PetscErrorCode TDyWYInitialize(DM dm,TDy tdy);
PETSC_EXTERN PetscErrorCode TDyWYComputeSystem(DM dm,TDy tdy,Mat K,Vec F);

PETSC_EXTERN PetscErrorCode TDyWYRecoverVelocity(DM dm,TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyWYPressureNorm(DM dm,TDy tdy,Vec U);
PETSC_EXTERN PetscReal TDyWYVelocityNorm(DM dm,TDy tdy);
PETSC_EXTERN PetscReal TDyWYDivergenceNorm(DM dm,TDy tdy);

/* ---------------------------------------------------------------- */

PETSC_EXTERN void PrintMatrix(PetscReal *A,PetscInt nr,PetscInt nc,PetscBool row_major);
PETSC_EXTERN PetscErrorCode CheckSymmetric(PetscReal *A,PetscInt n);

#endif
