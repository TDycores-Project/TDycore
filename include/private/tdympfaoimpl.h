#if !defined(TDYMPFAOIMPL_H)
#define TDYMPFAOIMPL_H

#include <tdycore.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdydiscretizationimpl.h>

// This struct stores MPFA-O specific data for the dycore.
typedef struct TDyMPFAO {
  // Options
  PetscInt gmatrix_method;
  PetscInt bc_type;
  PetscReal vangenuchten_m, vangenuchten_alpha;

  PetscReal mualem_poly_x0;
  PetscReal mualem_poly_x1;
  PetscReal mualem_poly_x2;
  PetscReal mualem_poly_dx;

  // TODO: Remove these when we're ready.
  // PetscBool output_geom_attributes;
  // PetscBool read_geom_attributes;
  // char geom_attributes_file[PETSC_MAX_PATH_LEN];

  // Mesh information
  TDyMesh *mesh;
  PetscReal *V; // cell volumes
  PetscReal *X; // point centroids
  PetscReal *N; // face normals
  PetscInt ncv, nfv; // number of {cell|face} vertices

  // Flow bookkeeping
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

  // For TH (here for now, may factor out)
  PetscReal ****Temp_subc_Gmatrix; // Gmatrix for subcells
  PetscReal ***Temp_Trans;
  Mat Temp_Trans_mat;
  Vec Temp_P_vec, Temp_TtimesP_vec;

  // For salinity (here for now, may factor out)
  PetscReal ****Psi_subc_Gmatrix; // Gmatrix for subcells
  PetscReal ***Psi_Trans;
  Mat Psi_Trans_mat;
  Vec Psi_vec, TtimesPsi_vec;
  PetscReal *m_nacl;

  //-----------------------------------
  // Material property data (per cell)
  //-----------------------------------
  PetscReal *K, *K0; // permeability
  PetscReal *porosity; // porosity
  PetscReal *Kappa, *Kappa0; // thermal conductivity
  PetscReal *rho_soil; // soil density
  PetscReal *c_soil; // soil specific heat
  PetscReal *D_saline; // saline diffusivity
  PetscReal *mu_saline; // saline molecular weight

  //--------------------------------------
  // Characteristic curve data (per cell)
  //--------------------------------------
  PetscReal *Kr, *dKr_dS;                 // relative permeability
  PetscReal *S, *dS_dP, *d2S_dP2, *dS_dT; // saturation
  PetscReal *Sr;                          // residual saturation

  // boundary pressure and dependent auxillary variables
  PetscReal *P_bnd;   // pressure [Pa]
  PetscReal *T_bnd;   // temperature [K]
  PetscReal *Psi_bnd; // salinity concentration [?]
  PetscReal *rho_bnd; // water density [kg m-3]
  PetscReal *vis_bnd; // water viscosity [Pa s]
  PetscReal *h_bnd;   // water enthalpy [?]

  PetscReal *Kr_bnd, *dKr_dS_bnd; // boundary rel perm and derivative
  PetscReal *S_bnd, *dS_dP_bnd, *d2S_dP2_bnd; // saturation and derivatives
  PetscReal *source_sink;          // flow equation source sink
  PetscReal *energy_source_sink;   // energy equation source sink
  PetscReal *salinity_source_sink; // salinity source sink

  //-----------------------------------------
  // non-linear functions of liquid pressure
  //-----------------------------------------

  // density of water [kg m-3]
  PetscReal  *rho, *drho_dP, *d2rho_dP2, *drho_dT;

  // viscosity of water [Pa s]
  PetscReal  *vis, *dvis_dP, *d2vis_dP2, *dvis_dT;

  // enthalpy of water [?]
  PetscReal  *h, *dh_dP, *dh_dT;

  // internal energy of water [?]
  PetscReal  *u, *du_dP, *du_dT;

  //-------------------
  // problem constants
  //-------------------
  PetscReal  gravity[3]; // vector of gravity [m s-2]
  PetscReal  Pref;       // reference pressure
  PetscReal  Tref;       // reference temperature

} TDyMPFAO;

// Functions specific to MPFA-O implementations.
PETSC_INTERN PetscErrorCode TDyCreate_MPFAO(void**);
PETSC_INTERN PetscErrorCode TDyDestroy_MPFAO(void*);
PETSC_INTERN PetscErrorCode TDySetFromOptions_MPFAO(void*, TDyOptions*);
PETSC_INTERN PetscErrorCode TDyComputeErrorNorms_MPFAO(void*,DM,Conditions*,Vec,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode TDyUpdateDiagnostics_MPFAO(void*,DM,Vec);

PETSC_INTERN PetscErrorCode TDyGetNumDMFields_Richards_MPFAO(void*);
PETSC_INTERN PetscErrorCode TDySetDMFields_Richards_MPFAO(void*, DM);
PETSC_INTERN PetscErrorCode TDySetup_Richards_MPFAO(void*, TDyDiscretizationType*, EOS*, MaterialProp*, CharacteristicCurves*, Conditions*);
PETSC_INTERN PetscErrorCode TDyUpdateState_Richards_MPFAO(void*, DM, EOS*, MaterialProp*, CharacteristicCurves*, PetscInt, PetscReal*);

PETSC_INTERN PetscErrorCode TDyGetNumDMFields_Richards_MPFAO_DAE(void*);
PETSC_INTERN PetscErrorCode TDySetDMFields_Richards_MPFAO_DAE(void*, DM);
PETSC_INTERN PetscErrorCode TDySetup_Richards_MPFAO_DAE(void*, TDyDiscretizationType*, EOS*, MaterialProp*, CharacteristicCurves*, Conditions*);

PETSC_INTERN PetscErrorCode TDySetDMFields_TH_MPFAO(void*, DM);
PETSC_INTERN PetscErrorCode TDySetup_TH_MPFAO(void*, TDyDiscretizationType*, EOS*, MaterialProp*, CharacteristicCurves*, Conditions*);
PETSC_INTERN PetscErrorCode TDyUpdateState_TH_MPFAO(void*, DM, EOS*, MaterialProp*, CharacteristicCurves*, PetscInt, PetscReal*);

PETSC_INTERN PetscErrorCode TDyGetNumDMFields_Salinity_MPFAO(void*);
PETSC_INTERN PetscErrorCode TDySetDMFields_Salinity_MPFAO(void*, DM);
PETSC_INTERN PetscErrorCode TDySetup_Salinity_MPFAO(void*, TDyDiscretizationType*, EOS*, MaterialProp*, CharacteristicCurves*, Conditions*);
PETSC_INTERN PetscErrorCode TDyUpdateState_Salinity_MPFAO(void*, DM, EOS*, MaterialProp*, CharacteristicCurves*, PetscInt, PetscReal*);

PETSC_INTERN PetscErrorCode TDyUpdateTransmissibilityMatrix(TDy);
PETSC_INTERN PetscErrorCode TDyComputeTransmissibilityMatrix(TDy);
PETSC_INTERN PetscErrorCode TDyComputeGravityDiscretization(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity(TDy,Vec);
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
PETSC_INTERN PetscErrorCode TDyMPFAOSNESFunction_Salinity(SNES,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESPreSolve(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAOSNESPreSolve_Salinity(TDy);

// Utils
PETSC_INTERN PetscErrorCode ExtractSubGmatrix(TDyMPFAO*,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode ExtractTempSubGmatrix(TDyMPFAO*,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode ExtractPsiSubGmatrix(TDyMPFAO*,PetscInt,PetscInt,PetscInt,PetscReal**);
PETSC_INTERN PetscErrorCode TDyMPFAOUpdateBoundaryState(TDy);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_InternalVertices(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_NotSharedWithInternalVertices(TDy, Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAORecoverVelocity_BoundaryVertices_SharedWithInternalVertices(TDy,Vec,PetscReal*,PetscInt*);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryPressure(TDy,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundaryTemperature(TDy,Vec);
PETSC_INTERN PetscErrorCode TDyMPFAO_SetBoundarySalinity(TDy,Vec);
PETSC_INTERN PetscErrorCode ComputeGtimesZ(PetscReal*,PetscReal*,PetscInt,PetscReal*);

#endif
