#if !defined(TDYWYIMPL_H)
#define TDYWYIMPL_H

#include <petsc.h>
#include <tdycore.h>
#include <private/tdyeosimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdydiscretizationimpl.h>

// This struct stores Wheeler-Yotov specific data for the dycore.
typedef struct TDyWY {
  // Options
  PetscReal vangenuchten_m, vangenuchten_alpha;
  PetscReal mualem_poly_x0;
  PetscReal mualem_poly_x1;
  PetscReal mualem_poly_x2;
  PetscReal mualem_poly_dx;

  // Reference pressure.
  PetscReal Pref;

  // Mesh information
  PetscInt dim; // spatial dimension
  PetscReal *V; // cell volumes
  PetscReal *X; // point centroids
  PetscReal *N; // face normals
  PetscInt ncv, nfv; // number of {cell|face} vertices

  PetscInt  *vmap;      /* [cell,local_vertex] --> global_vertex */
  PetscInt  *emap;      /* [cell,local_vertex,direction] --> global_face */
  PetscInt  *fmap;      /* [face,local_vertex] --> global_vertex */
  PetscReal *Alocal;    /* local element matrices (Ku,v) */
  PetscReal *Flocal;    /* local element vectors (f,w) */
  PetscQuadrature quad; /* vertex-based quadrature rule */
  PetscReal *vel;       /* [face,local_vertex] --> velocity normal to face at vertex */
  PetscInt *vel_count;  /* For MPFAO, the number of subfaces that are used to determine velocity at the face. For 3D+hex, vel_count = 4 */

  PetscInt  *faces;

  // Quadrature type.
  TDyQuadratureType qtype;

  // Material property data
  PetscReal *K, *K0; // permeability
  PetscReal *porosity;

  PetscReal *Kr, *dKr_dS; // relative permeability and derivative per cell
  PetscReal *S, *dS_dP, *d2S_dP2; // saturation and derivatives per cell
  PetscReal *Sr; // residual saturation
} TDyWY;

// Functions specific to WY implementation
PETSC_INTERN PetscErrorCode TDyCreate_WY(void**);
PETSC_INTERN PetscErrorCode TDyDestroy_WY(void*);
PETSC_INTERN PetscErrorCode TDySetFromOptions_WY(void*, TDyOptions*);
PETSC_INTERN PetscErrorCode TDySetDMFields_WY(void*,DM);
PETSC_INTERN PetscInt TDyGetNumDMFields_WY(void*);
PETSC_INTERN PetscErrorCode TDySetup_WY(void*,TDyDiscretizationType*,EOS*,MaterialProp*,CharacteristicCurves*,Conditions*);
PETSC_INTERN PetscErrorCode TDyUpdateState_WY(void*,DM,EOS*,MaterialProp*,CharacteristicCurves*,PetscInt,PetscReal*);

PETSC_INTERN PetscErrorCode TDyWYRecoverVelocity(TDy,Vec);
// TODO: This should be internal.
PETSC_EXTERN PetscErrorCode TDyWYResidual(TS,PetscReal,Vec,Vec,Vec,void *ctx);
PETSC_INTERN PetscErrorCode Pullback(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar,PetscInt);
//PETSC_INTERN PetscErrorCode IntegrateOnFace(TDy,PetscInt,PetscInt,PetscReal*);

#endif
