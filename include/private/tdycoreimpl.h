#if !defined(TDYCOREIMPL_H)
#define TDYCOREIMPL_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyregressionimpl.h>
#include <tdycore.h>
#include <tdyio.h>
#include <private/tdytiimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdyconditionsimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdyoptions.h>

#define VAR_PRESSURE 0
#define VAR_TEMPERATURE 1

// This type serves as a "virtual table" containing function pointers that
// define the behavior of the dycore.
typedef struct _TDyOps *TDyOps;
struct _TDyOps {
  // Called by TDyCreate to allocate implementation-specific resources. Returns
  // a pointer to a context.
  PetscErrorCode (*create)(TDy);

  // Called by TDyDestroy to free implementation-specific resources.
  PetscErrorCode (*destroy)(void*);

  // Creates a DM for a particular simulation (optional).
  // Arguments:
  //   1. A valid MPI communicator
  //   2. A context pointer storing data specifically for the constructor
  //      function
  //   3. A location for the newly created DM.
  // We pass the dycore as the first argument here because we don't expect
  // the caller to know implementation details.
  PetscErrorCode (*create_dm)(MPI_Comm, void*, DM*);

  // Implements the view operation for the TDy implementation with the given
  // viewer.
  PetscErrorCode (*view)(void*, PetscViewer);

  // Called by TDySetFromOptions -- sets implementation-specific options from
  // command-line arguments.
  PetscErrorCode (*set_from_options)(void*);

  // Called by TDySetup -- configures the DM for solvers.
  PetscErrorCode (*setup)(void*, DM, MaterialProp*, TDyConditions*);

  // Called by TDyComputeErrorNorms -- computes error norms given a solution
  // vector.
  PetscErrorCode (*compute_error_norms)(void*,Vec,PetscReal*,PetscReal*);

  // Functions used to define solver behavior.

  // Material and boundary condition functions--we'll sort these out later.
  PetscErrorCode (*computeporosity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computepermeability)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computethermalconductivity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeresidualsaturation)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computesoildensity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computesoilspecificheat)(TDy,PetscReal*,PetscReal*,void*);
};

// This type represents the dycore and all of its settings.
struct _p_TDy {
  PETSCHEADER(struct _TDyOps);

  // Implementation-specific context pointer
  void *context;

  // Flags that indicate where the dycore is in the setup process
  TDySetupFlags setup_flags;

  // Grid and data management -- handed to a solver when the dycore is fully
  // configured
  DM dm;

  // Contextual information passed to create_dm (if given).
  void* create_dm_context;

  // I/O subsystem
  TDyIO io;

  // options that determine the behavior(s) of the dycore
  TDyOptions options;

  // boundary conditions and sources/sinks
  TDyConditions conditions;

  // regression testing data
  TDyRegression *regression;

  //---------------------------------------------------
  // Material models (probably should be factored out)
  //---------------------------------------------------

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

  /* material parameters */
  MaterialProp *matprop;

  /* characteristic curve parameters */
  CharacteristicCurve *cc;

  //------------------------------------------------------
  // Solver-specific information (should be factored out)
  //------------------------------------------------------
  Mat J, Jpre;

  TDyTimeIntegrator ti;

  /* For SNES based timestepping */
  PetscReal dtime;
  Vec solution;
  Vec soln_prev;
  Vec accumulation_prev;
  Vec residual;

};



#endif
