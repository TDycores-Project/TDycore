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
  PetscErrorCode (*computepermeability)(TDy,PetscReal*,PetscReal*,void*);
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

  /* problem constants */
  PetscReal  rho;        /* density of water [kg m-3]*/
  PetscReal  mu;         /* viscosity of water [Pa s] */
  PetscReal  Sr;         /* residual saturation (min) [1] */
  PetscReal  Ss;         /* saturated saturation (max) [1] */
  PetscReal  gravity[3]; /* vector of gravity [m s-2] */
  PetscReal  Pref;       /* reference pressure */


  /* material parameters */
  PetscReal *K,
            *K0;     /* permeability tensor (cell,intrinsic) for each cell [m2] */
  PetscReal *Kr;        /* relative permeability for each cell [1] */
  PetscReal *porosity;  /* porosity for each cell [1] */
  PetscReal *S,
            *dS_dP;  /* saturation and derivative wrt pressure for each cell [1] */

  void *permeabilityctx;
  void *forcingctx;
  void *dirichletvaluectx;
  void *dirichletfluxctx;

  SpatialFunction forcing;
  SpatialFunction dirichlet;
  SpatialFunction flux;

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

  PetscInt  *LtoG;
  PetscInt  *orient;
  PetscInt  *faces;

  /* MPFA-O */
  TDy_mesh *mesh;
  PetscReal ****subc_Gmatrix; /* Gmatrix for subcells */
  PetscReal ***Trans;

  PetscBool regression_testing;
  TDy_regression *regression;
  

};



#endif
