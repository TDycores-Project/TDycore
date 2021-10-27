#if !defined(TDYMPFAOIMPL_H)
#define TDYMPFAOIMPL_H

#include <tdycore.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdymeshimpl.h>

// This struct stores MPFA-O specific data for the dycore.
typedef struct TDyMPFAO {
  // Options
  PetscInt gmatrix_method;
  PetscInt bc_type;

  // Mesh information
  TDyMesh *mesh;
  PetscReal *V; // cell volumes
  PetscReal *X; // point centroids
  PetscReal *N; // face normals
  PetscInt *closureSize, **closure, maxClosureSize;
  PetscInt ncv, nfv; // number of {cell|face} vertices

  PetscReal ****subc_Gmatrix; // Gmatrix for subcells
  PetscReal ***Trans;
  Mat Trans_mat;
  Vec P_vec, TtimesP_vec;
  Vec GravDisVec;

  // [face,local_vertex] --> velocity normal to face at vertex
  PetscReal *vel;
  // For MPFAO, the number of subfaces that are used to determine velocity at
  // the face. For 3D+hex, vel_count = 4
  PetscInt *vel_count;

  // For temperature -- here for now, may factor this out further.
  PetscReal ****Temp_subc_Gmatrix; // Gmatrix for subcells
  PetscReal ***Temp_Trans;
  Mat Temp_Trans_mat;
  Vec Temp_P_vec, Temp_TtimesP_vec;

  /* boundary pressure and auxillary variables that depend on boundary pressure */
  PetscReal *P_BND;
  PetscReal *T_BND;              /* boundary temperature */
  PetscReal *rho_BND;            /* density of water [kg m-3]*/
  PetscReal *vis_BND;            /* viscosity of water [Pa s] */
  PetscReal *h_BND;              /* enthalpy of water */

  CharacteristicCurve *cc_bnd;
  PetscReal *Kr_BND; /* relative permeability for each cell [1] */
  PetscReal *S_BND;  /* saturation, first derivative wrt boundary pressure, and */
  PetscReal *source_sink;         /* flow equation source sink */
  PetscReal *energy_source_sink;  /* energy equation source sink */

  /* non-linear function of liquid pressure */
  PetscReal  *rho, *drho_dP, *d2rho_dP2;       /* density of water [kg m-3]*/
  PetscReal  *vis, *dvis_dP, *d2vis_dP2;       /* viscosity of water [Pa s] */
  PetscReal  *h, *dh_dP, *dh_dT;               /* enthalpy of water */
  PetscReal  *u, *du_dP, *du_dT;               /* internal energy of water */
  PetscReal  *drho_dT, *dvis_dT;

  /* problem constants */
  PetscReal  gravity[3]; /* vector of gravity [m s-2] */
  PetscReal  Pref;       /* reference pressure */
  PetscReal  Tref;       /* reference temperature */

} TDyMPFAO;

// Functions specific to MPFA-O implementations.
PETSC_INTERN PetscErrorCode TDyCreate_MPFAO(void**);
PETSC_INTERN PetscErrorCode TDyDestroy_MPFAO(void*);
PETSC_INTERN PetscErrorCode TDySetFromOptions_MPFAO(void*);
PETSC_INTERN PetscErrorCode TDySetup_Richards_MPFAO(void*, DM, MaterialProp*, TDyConditions*);
PETSC_INTERN PetscErrorCode TDySetup_Richards_MPFAO_DAE(void*, DM, MaterialProp*, TDyConditions*);
PETSC_INTERN PetscErrorCode TDySetup_Richards_MPFAO_TRANSIENTVAR(void*, DM, MaterialProp*, TDyConditions*);
PETSC_INTERN PetscErrorCode TDySetup_TH_MPFAO(void*, DM, MaterialProp*, TDyConditions*);
PETSC_INTERN PetscErrorCode TDyUpdateState_Richards_MPFAO(void*, DM, MaterialProp*, CharacteristicCurve*);
PETSC_INTERN PetscErrorCode TDyUpdateState_TH_MPFAO(void*, DM, MaterialProp*, CharacteristicCurve*);
PETSC_INTERN PetscErrorCode TDyUpdateTransmissibilityMatrix(TDy);
PETSC_INTERN PetscErrorCode TDyComputeTransmissibilityMatrix(TDy);
PETSC_INTERN PetscErrorCode TDyComputeGravityDiscretization(TDy);
PETSC_EXTERN PetscErrorCode TDyMPFAOComputeSystem(TDy,Mat,Vec);
PETSC_EXTERN PetscErrorCode TDyMPFAORecoverVelocity(TDy,Vec);
PETSC_EXTERN PetscReal TDyMPFAOVelocityNorm(TDy);
PETSC_EXTERN PetscReal TDyMPFAOPressureNorm(TDy,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_InternalVertices(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_SharedWithInternalVertices(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOComputeSystem_BoundaryVertices_NotSharedWithInternalVertices(TDy,Mat,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian_TH(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_TH(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_DAE(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOTransientVariable(TS,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOIFunction_TransientVariable(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESFunction(SNES,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESJacobian(SNES,Vec,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESPreSolve(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOSetGmatrixMethod(TDyMPFAO*,TDyMPFAOGmatrixMethod);
PETSC_INTERN PetscErrorCode TDyMPFAOSetBoundaryConditionType(TDyMPFAO*,TDyMPFAOBoundaryConditionType);

#endif
