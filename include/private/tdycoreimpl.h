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

#define VAR_PRESSURE 0
#define VAR_TEMPERATURE 1

typedef struct _TDyOps *TDyOps;
struct _TDyOps {
  PetscErrorCode (*create)(TDy);
  PetscErrorCode (*destroy)(TDy);
  PetscErrorCode (*view)(TDy);
  PetscErrorCode (*setup)(TDy);
  PetscErrorCode (*setfromoptions)(TDy);
  PetscErrorCode (*computeporosity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computepermeability)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computethermalconductivity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeresidualsaturation)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeforcing)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeenergyforcing)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computedirichletvalue)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computetemperaturedirichletvalue)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computedirichletflux)(TDy,PetscReal*,PetscReal*,void*);
};

struct _p_TDy {
  PETSCHEADER(struct _TDyOps);
  PetscBool setup;
  DM dm;

  TDyTimeIntegrator ti;
  TDySetupFlags setupflags;
  TDyIO io;
  PetscBool init_with_random_field;

  /* arrays of the size of the Hasse diagram */
  PetscReal *V; /* volume of point (if applicable) */
  PetscReal *X; /* centroid of point */
  PetscReal *N; /* normal of point (if applicable) */
  PetscInt ncv,nfv; /* number of {cell|face} vertices */

  /* non-linear function of liquid pressure */
  PetscInt rho_type;
  PetscInt mu_type;
  PetscInt enthalpy_type;
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
  PetscReal *matprop_K,
            *matprop_K0;         /* permeability tensor (cell,intrinsic) for each cell [m2] */
  PetscReal *matprop_porosity;   /* porosity for each cell [1] */
  PetscReal *matprop_Kappa,
            *matprop_Kappa0;     /* thermal conductivity tensor (cell,intrinsic) for each cell [W/(K-m)] */
  PetscReal *matprop_Cr;         /* specific heat capacity for rock [J/(kg-K)] */
  PetscReal *matprop_rhor;       /* rock density [kg/m3] */

/* characteristic curver parameters */
  CharacteristicCurve cc;
  PetscReal *cc_m;               /* parameter used in saturation and relative permeability function [-]*/
  PetscReal *cc_n;               /* parameter used in Gardner saturation function [-] */
  PetscReal *cc_alpha;           /* parameter used in VanGenuchten saturation function [-] */
  PetscReal *S,                  /* saturation for each cell */
            *dS_dP,              /* derivative of saturation wrt pressure for each cell [Pa^-1] */
            *d2S_dP2,            /* second derivative of saturation wrt pressure for each cell [Pa^-2] */
            *dS_dT;              /* derivate of saturation wrt to temperature for each cell [K^-1] */
  PetscReal *Kr, *dKr_dS;        /* relative permeability for each cell [1] */

  /* boundary pressure and auxillary variables that depend on boundary pressure */
  PetscReal *P_BND;
  PetscReal *T_BND;               /* boundary temperature */
  PetscReal  *rho_BND;            /* density of water [kg m-3]*/
  PetscReal  *vis_BND;            /* viscosity of water [Pa s] */
  PetscReal  *h_BND;              /* enthalpy of water */
  PetscReal *Kr_BND, *dKr_dS_BND; /* relative permeability for each cell [1] */
  PetscReal *S_BND,  *dS_dP_BND,  /* saturation, first derivative wrt boundary pressure, and */
            *d2S_dP2_BND;         /* second derivative of saturation wrt boundary pressure */
  PetscReal *source_sink;         /* flow equation source sink */
  PetscReal *energy_source_sink;  /* energy equation source sink */

  void *porosityctx;
  void *permeabilityctx;
  void *thermalconductivityctx;
  void *residualsaturationctx;
  void *forcingctx;
  void *energyforcingctx;
  void *dirichletvaluectx;
  void *temperaturedirichletvaluectx;
  void *dirichletfluxctx;

  /* method-specific information*/
  TDyMethod method;
  TDyMode mode;
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
  PetscInt *vel_count;  /* For MPFAO, the number of subfaces that are used to determine velocity at the face. For 3D+hex, vel_count = 4 */

  PetscInt  *LtoG;
  PetscInt  *orient;
  PetscInt  *faces;

  /* MPFA-O */
  PetscInt mpfao_gmatrix_method;
  PetscInt mpfao_bc_type;

  TDy_mesh *mesh;
  PetscReal ****subc_Gmatrix; /* Gmatrix for subcells */
  PetscReal ***Trans;
  Mat Trans_mat;
  Vec P_vec, TtimesP_vec;

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

  PetscBool output_mesh;
  PetscBool regression_testing;
  TDy_regression *regression;

  /* Timers enabled? */
  PetscBool enable_timers;
};



#endif
