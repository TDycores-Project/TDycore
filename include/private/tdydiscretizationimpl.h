#if !defined(TDYDISCRETIZATION_H)
#define TDYDISCRETIZATION_H

#include <petsc.h>
#include <private/tdydmimpl.h>


typedef struct {

  TDyDM tdydm;
  TDyUGrid ugrid;

} TDyDiscretizationType;


PETSC_INTERN PetscErrorCode TDyDiscretizationCreate(TDyDiscretizationType*);
PETSC_INTERN PetscErrorCode TDyDiscretizationCreateFromPFLOTRANMesh(const char*,PetscInt,TDyDiscretizationType*);
PETSC_INTERN PetscErrorCode TDyDiscretizationCreateGlobalVector(TDyDiscretizationType*,Vec*);
PETSC_INTERN PetscErrorCode TDyDiscretizationCreateLocalVector(TDyDiscretizationType*,Vec*);
PETSC_INTERN PetscErrorCode TDyDiscretizationCreateNaturalVector(TDyDiscretizationType*,Vec*);
PETSC_INTERN PetscErrorCode TDyDiscretizationCreateJacobianMatrix(TDyDiscretizationType*,Mat*);
PETSC_INTERN PetscErrorCode TDyDiscretizationGlobalToNatural(TDyDiscretizationType*,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyDiscretizationGlobalToLocal(TDyDiscretizationType*,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyDiscretizationNaturalToGlobal(TDyDiscretizationType*,Vec,Vec);
PETSC_INTERN PetscErrorCode TDyDiscretizationNaturaltoLocal(TDyDiscretizationType*,Vec,Vec*);
PETSC_INTERN PetscErrorCode TDyDiscretizationGetTDyDM(TDyDiscretizationType*,TDyDM*);
PETSC_INTERN PetscErrorCode TDyDiscretizationGetDM(TDyDiscretizationType*,DM*);


#endif

