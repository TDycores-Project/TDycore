#if !defined(TDYDMIMPL_H)
#define TDYDMIMPL_H

#include <petsc.h>
#include <private/tdyugdmimpl.h>
#include <private/tdyugridimpl.h>

typedef enum{
  PLEX_TYPE=0,    /* PETSc's DMPlex type*/
  TDYCORE_DM_TYPE /* TDycore-managed DMShell*/
} TDyDMType;


// A struct that is a wrapper for PETSc DM.
// - Typically, DM is a PETSc DMPlex
// - When an unstructured grid mesh in TDycore's format is used, DM is a DMShell.
//   First, a TDycore-specific unstructured grid DM is created. Then, a DMShell
//   is created.
//   NOTE: The format of unstructured grid used in TDycore is same as the one used in
//   PFLOTRAN.
typedef struct {
  DM dm;
  TDyUGDM *ugdm;

  TDyDMType dmtype;

} TDyDM;

PETSC_INTERN PetscErrorCode TDyDMCreate(TDyDM**);
PETSC_INTERN PetscErrorCode TDyDMDestroy(TDyDM*);
PETSC_INTERN PetscErrorCode TDyDMCreateFromUGrid(PetscInt,TDyUGrid*,TDyDM*);

#endif
