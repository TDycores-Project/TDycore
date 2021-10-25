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
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdyoptions.h>

#define VAR_PRESSURE 0
#define VAR_TEMPERATURE 1

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

  // Called by TDySetFromOptions -- sets implementation-specific options
  // from command-line arguments.
  // FIXME: convert the arg here to void* when we've moved specific data
  // FIXME: out of TDy.
  PetscErrorCode (*set_from_options)(TDy);

  // Called by TDySetup -- configures the DM for solvers.
  // FIXME: we should convert the first argument here to void* when we've moved
  // FIXME: all discretization-specific data out of TDy itself.
  PetscErrorCode (*setup)(TDy, DM);

  // Called by TDyComputeErrorNorms -- computes error norms given a solution
  // vector.
  PetscErrorCode (*compute_error_norms)(void*,Vec,PetscReal*,PetscReal*);

  // Functions used to define solver behavior.

  // Material and boundary condition functions--we'll sort these out later.
  PetscErrorCode (*computeporosity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computepermeability)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computethermalconductivity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeresidualsaturation)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeforcing)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeenergyforcing)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*compute_boundary_pressure)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*compute_boundary_temperature)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*compute_boundary_velocity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computesoildensity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computesoilspecificheat)(TDy,PetscReal*,PetscReal*,void*);
};

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

  // We'll likely get rid of this.
  TDyTimeIntegrator ti;

  // I/O subsystem
  TDyIO io;

  // options that determine the behavior(s) of the dycore
  TDyOptions options;

  /* arrays of the size of the Hasse diagram */
  PetscReal *V; /* volume of point (if applicable) */
  PetscReal *X; /* centroid of point */
  PetscReal *N; /* normal of point (if applicable) */
  PetscInt ncv,nfv; /* number of {cell|face} vertices */

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

  /* boundary pressure and auxillary variables that depend on boundary pressure */
  PetscReal *P_BND;
  PetscReal *T_BND;               /* boundary temperature */
  PetscReal  *rho_BND;            /* density of water [kg m-3]*/
  PetscReal  *vis_BND;            /* viscosity of water [Pa s] */
  PetscReal  *h_BND;              /* enthalpy of water */

  CharacteristicCurve *cc_bnd;
  PetscReal *Kr_BND; /* relative permeability for each cell [1] */
  PetscReal *S_BND;  /* saturation, first derivative wrt boundary pressure, and */
  PetscReal *source_sink;         /* flow equation source sink */
  PetscReal *energy_source_sink;  /* energy equation source sink */

  void *porosityctx;
  void *permeabilityctx;
  void *thermalconductivityctx;
  void *residualsaturationctx;
  void *forcingctx;
  void *energyforcingctx;
  void *boundary_pressure_ctx;
  void *boundary_temperature_ctx;
  void *boundary_velocity_ctx;
  void *soildensityctx;
  void *soilspecificheatctx;

  /* Wheeler-Yotov */
  PetscInt  *vmap;      /* [cell,local_vertex] --> global_vertex */
  PetscInt  *emap;      /* [cell,local_vertex,direction] --> global_face */
  PetscInt  *fmap;      /* [face,local_vertex] --> global_vertex */
  PetscReal *Alocal;    /* local element matrices (Ku,v) */
  PetscReal *Flocal;    /* local element vectors (f,w) */
  PetscQuadrature quad; /* vertex-based quadrature rule */
  PetscReal *vel;       /* [face,local_vertex] --> velocity normal to face at vertex */
  PetscInt *vel_count;  /* For MPFAO, the number of subfaces that are used to determine velocity at the face. For 3D+hex, vel_count = 4 */

  PetscInt  *LtoG;
  PetscInt  *orient;
  PetscInt  *faces;

  /* MPFA-O */
  TDyMesh *mesh;
  PetscReal ****subc_Gmatrix; /* Gmatrix for subcells */
  PetscReal ***Trans;
  Mat Trans_mat;
  Vec P_vec, TtimesP_vec;
  Vec GravDisVec;

  /* For temperature */
  PetscReal ****Temp_subc_Gmatrix; /* Gmatrix for subcells */
  PetscReal ***Temp_Trans;
  Mat Temp_Trans_mat;
  Vec Temp_P_vec, Temp_TtimesP_vec;

  Mat J, Jpre;

  /* For SNES based timestepping */
  PetscReal dtime;
  Vec solution;
  Vec soln_prev;
  Vec accumulation_prev;
  Vec residual;

  PetscInt *closureSize, **closure, maxClosureSize;

  TDyRegression *regression;
};



#endif
