#if !defined(TDYIOIMPL_H)
#define TDYIOIMPL_H

#include <petsc.h>

struct _p_TDyIO {
  PetscBool io_process;
  PetscBool print_intermediate;
  char filename[PETSC_MAX_PATH_LEN];
  char zonalVarNames[2][PETSC_MAX_PATH_LEN];
  int num_vars;
  TDyIOFormat format;
  int num_times;
};

PETSC_INTERN PetscErrorCode TDyIOCreate(TDyIO*);
PETSC_INTERN PetscErrorCode TDyIOInitializeExodus(char*,char*[],DM ,int);
PETSC_INTERN PetscErrorCode TDyIOAddExodusTime(char*,PetscReal,DM,TDyIO);
PETSC_INTERN PetscErrorCode TDyIOWriteExodusVar(char*,Vec,char*,TDyIO,PetscReal);
PETSC_EXTERN PetscErrorCode TDyIOInitializeHDF5(char*,DM);
PETSC_EXTERN PetscErrorCode TDyIOWriteHDF5Var(char*,DM,Vec,char*,PetscReal);
PETSC_EXTERN PetscErrorCode TDyIOWriteAsciiViewer(Vec,PetscReal,char*);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFHeader(PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFAttribute(char*,char*,PetscInt);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFFooter();
PETSC_INTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif

