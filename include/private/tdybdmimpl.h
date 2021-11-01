#if !defined(TDYBDMIMPL_H)
#define TDYBDMIMPL_H

#include <petsc.h>
#include <tdycore.h>

// This struct stores BDM specific data for the dycore.
typedef struct TDyBDM {

  // Mesh information
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

  PetscInt  *LtoG;
  PetscInt  *orient;
  PetscInt  *faces;

  // Quadrature type.
  TDyQuadratureType qtype;
} TDyBDM;

// Functions specific to BDM implementation
PETSC_INTERN PetscErrorCode TDyCreate_BDM(void**);
PETSC_INTERN PetscErrorCode TDyDestroy_BDM(void*);
PETSC_INTERN PetscErrorCode TDySetFromOptions_BDM(void*,TDyOptions*);
PETSC_INTERN PetscErrorCode TDySetup_BDM(void*,DM,EOS*,MaterialProp*,CharacteristicCurves*,Conditions*);
PETSC_INTERN PetscReal TDyBDMPressureNorm(TDy,Vec);
PETSC_INTERN PetscReal TDyBDMVelocityNorm(TDy,Vec);

#endif
