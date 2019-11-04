#if !defined(TDYCOREIMPL_H)
#define TDYCOREIMPL_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyregressionimpl.h>
#include <tdycore.h>

typedef struct _TDyOps *TDyOps;
struct _TDyOps {
  PetscErrorCode (*create)(TDy);
  PetscErrorCode (*destroy)(TDy);
  PetscErrorCode (*view)(TDy);
  PetscErrorCode (*setup)(TDy);
  PetscErrorCode (*setfromoptions)(TDy);
  PetscErrorCode (*computeporosity)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computepermeability)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeresidualsaturation)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computeforcing)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computedirichletvalue)(TDy,PetscReal*,PetscReal*,void*);
  PetscErrorCode (*computedirichletflux)(TDy,PetscReal*,PetscReal*,void*);
};

struct _p_TDy {
  PETSCHEADER(struct _TDyOps);
  PetscBool setup;
  DM dm;

  /* arrays of the size of the Hasse diagram */
  PetscReal *V; /* volume of point (if applicable) */
  PetscReal *X; /* centroid of point */
  PetscReal *N; /* normal of point (if applicable) */
  PetscInt ncv,nfv; /* number of {cell|face} vertices */
  
  /* non-linear function of liquid pressure */
  PetscInt rho_type;
  PetscInt mu_type;
  PetscReal  *rho, *drho_dP, *d2rho_dP2;       /* density of water [kg m-3]*/
  PetscReal  *vis, *dvis_dP, *d2vis_dP2;       /* viscosity of water [Pa s] */

  /* problem constants */
  PetscReal  gravity[3]; /* vector of gravity [m s-2] */
  PetscReal  Pref;       /* reference pressure */


  /* material parameters */
  PetscReal  *Sr;        /* residual saturation (min) [1] */
  PetscReal *K,
            *K0;     /* permeability tensor (cell,intrinsic) for each cell [m2] */
  PetscReal *Kr, *dKr_dS;        /* relative permeability for each cell [1] */
  PetscReal *porosity;  /* porosity for each cell [1] */
  PetscReal *S,
            *dS_dP,  /* saturation and derivative wrt pressure for each cell [1] */
            *d2S_dP2;/* second derivative of saturation wrt pressure for each cell [1] */

  PetscInt *SatFuncType;     /* type of saturation function */
  PetscInt *RelPermFuncType; /* type of relative permeability */

  PetscReal *matprop_m, *matprop_n, *matprop_alpha;

  /* boundary pressure and auxillary variables that depend on boundary pressure */
  PetscReal *P_BND;
  PetscReal  *rho_BND;            /* density of water [kg m-3]*/
  PetscReal  *vis_BND;             /* viscosity of water [Pa s] */
  PetscReal *Kr_BND, *dKr_dS_BND; /* relative permeability for each cell [1] */
  PetscReal *S_BND,  *dS_dP_BND,  /* saturation, first derivative wrt boundary pressure, and */
            *d2S_dP2_BND;         /* second derivative of saturation wrt boundary pressure */
  PetscReal *source_sink;

  void *porosityctx;
  void *permeabilityctx;
  void *residualsaturationctx;
  void *forcingctx;
  void *dirichletvaluectx;
  void *dirichletfluxctx;

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
  PetscInt *vel_count;  /* For MPFAO, the number of subfaces that are used to determine velocity at the face. For 3D+hex, vel_count = 4 */

  PetscInt  *LtoG;
  PetscInt  *orient;
  PetscInt  *faces;

  /* MPFA-O */
  TDy_mesh *mesh;
  PetscReal ****subc_Gmatrix; /* Gmatrix for subcells */
  PetscReal ***Trans;

  Mat J, Jpre;

  PetscInt *closureSize, **closure, maxClosureSize;

  PetscBool output_mesh;
  PetscBool regression_testing;
  TDy_regression *regression;
  

};



#endif
