#if !defined(TDYFVTPFIMPL_H)
#define TDYFVTPFIMPL_H

#include <tdycore.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdyeosimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdymeshimpl.h>

// This struct stores FV-TPF specific data for the dycore.
typedef struct TDyFVTPF {
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

  PetscReal *pressure;

  // [face,local_vertex] --> velocity normal to face at vertex
  PetscReal *vel;
  // For FV-TPF, the number of subfaces that are used to determine velocity at
  // the face. For 3D+hex, vel_count = 4
  PetscInt *vel_count;

  // Material property data
  PetscReal *K, *K0; // permeability per cell
  PetscReal *porosity; // porosity per cell
  PetscReal *Kappa, *Kappa0; // thermal conductivity per cell
  PetscReal *rho_soil; // soil density per cell
  PetscReal *c_soil; // soil specific heat per cell

  // Characteristic curve data
  PetscReal *Kr, *dKr_dS; // relative permeability and derivative per cell
  PetscReal *S, *dS_dP, *d2S_dP2, *dS_dT; // saturation and derivatives per cell
  PetscReal *Sr; // residual saturation

  // boundary pressure and auxillary variables that depend on boundary pressure
  PetscReal *P_bnd; // pressure [Pa]
  PetscReal *T_bnd; // temperature [K]
  PetscReal *rho_bnd; // water density [kg m-3]
  PetscReal *vis_bnd; // water viscosity [Pa s]
  PetscReal *h_bnd; // water enthalpy

  PetscReal *Kr_bnd, *dKr_dS_bnd; // boundary rel perm and derivative
  PetscReal *S_bnd, *dS_dP_bnd, *d2S_dP2_bnd; // saturation and derivatives
  PetscReal *source_sink;         // flow equation source sink
  PetscReal *energy_source_sink;  // energy equation source sink

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

} TDyFVTPF;

// Functions specific to FVTPF implementations.
PETSC_INTERN PetscErrorCode TDyCreate_FVTPF(void**);
PETSC_INTERN PetscErrorCode TDyDestroy_FVTPF(void*);
PETSC_INTERN PetscErrorCode TDySetFromOptions_FVTPF(void*, TDyOptions*);
PETSC_INTERN PetscErrorCode TDyGetNumDMFields_Richards_FVTPF(void*);
PETSC_INTERN PetscErrorCode TDySetDMFields_Richards_FVTPF(void*, DM);
PETSC_INTERN PetscErrorCode TDySetup_Richards_FVTPF(void*, TDyDiscretizationType*, EOS*, MaterialProp*, CharacteristicCurves*, Conditions*);
PETSC_INTERN PetscErrorCode TDyUpdateState_Richards_FVTPF(void*, DM, EOS*, MaterialProp*, CharacteristicCurves*, PetscInt, PetscReal*);
PETSC_INTERN PetscErrorCode TDyUpdateDiagnostics_FVTPF(void*,DM,Vec);

PETSC_INTERN PetscErrorCode TDyFVTPFSNESFunction(SNES,Vec,Vec,void*);
PETSC_INTERN PetscErrorCode TDyFVTPFSNESJacobian(SNES,Vec,Mat,Mat,void*);
PETSC_INTERN PetscErrorCode TDyFVTPFSNESPreSolve(TDy);
PETSC_INTERN PetscErrorCode TDyFVTPFSNESTimeCut(TDy);

// Utils
PETSC_INTERN PetscErrorCode TDyFVTPFUpdateBoundaryState(TDy);
PETSC_INTERN PetscErrorCode TDyFVTPFSetBoundaryPressure(TDy,Vec);

PETSC_INTERN PetscErrorCode FVTPFComputeFacePermeabililtyValueTPF(TDyFVTPF*, MaterialProp*, PetscInt, PetscInt, PetscReal*, PetscReal*);
PETSC_INTERN PetscErrorCode FVTPFCalculateDistances(TDyFVTPF*, PetscInt, PetscInt, PetscReal*, PetscReal*);

#endif
