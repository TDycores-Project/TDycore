#if !defined(TDYBDMIMPL_H)
#define TDYBDMIMPL_H

#include <petsc.h>
#include <tdycore.h>
#include <private/tdydiscretizationimpl.h>

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
  PetscQuadrature quad; /* vertex-based quadrature rule */

  PetscInt  *LtoG;
  PetscInt  *orient;
  PetscInt  *faces;

  // Quadrature type.
  TDyQuadratureType qtype;

  // Material property data
  PetscReal *K; // permeability
} TDyBDM;

// Functions specific to BDM implementation
PETSC_INTERN PetscErrorCode TDyCreate_BDM(void**);
PETSC_INTERN PetscErrorCode TDyDestroy_BDM(void*);
PETSC_INTERN PetscErrorCode TDySetFromOptions_BDM(void*,TDyOptions*);
PETSC_INTERN PetscErrorCode TDySetDMFields_BDM(void*,DM);
PETSC_INTERN PetscInt TDyGetNumDMFields_BDM(void*);
PETSC_INTERN PetscErrorCode TDySetup_BDM(void*,TDyDiscretizationType*,EOS*,MaterialProp*,CharacteristicCurves*,Conditions*);

#endif
