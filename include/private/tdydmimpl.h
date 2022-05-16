#if !defined(TDYDMIMPL_H)
#define TDYDMIMPL_H

#include <petsc.h>
#include <private/tdyugdmimpl.h>
#include <private/tdyugridimpl.h>

typedef enum{
  PLEX_TYPE=0,
  TDYCORE_DM_TYPE
} TDyDMType;

typedef struct {
  DM dm;
  TDyUGDM *ugdm;

  TDyDMType dmtype;

} TDyDM;

PETSC_INTERN PetscErrorCode TDyDMCreate(TDyDM**);
PETSC_INTERN PetscErrorCode TDyDMDestroy(TDyDM*);
PETSC_INTERN PetscErrorCode TDyDMCreateFromUGrid(PetscInt,TDyUGrid*,TDyDM*);

#endif