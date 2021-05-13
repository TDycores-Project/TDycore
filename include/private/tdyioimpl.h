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
PETSC_INTERN PetscErrorCode TdyIOInitializeExodus(char*,char*[],DM ,int);
PETSC_INTERN PetscErrorCode TdyIOAddExodusTime(char*,PetscReal,TDyIO);
PETSC_INTERN PetscErrorCode TdyIOWriteExodusVar(char*,DM,Vec,Vec,TDyIO,PetscReal);
PETSC_EXTERN PetscErrorCode TdyIOInitializeHDF5(char*,DM);
PETSC_EXTERN PetscErrorCode TdyIOWriteHDF5Var(char*,DM,Vec,Vec,PetscReal);
PETSC_EXTERN PetscErrorCode TDyIOWriteAsciiViewer(Vec,Vec,PetscReal);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFHeader(PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFAttribute(char*,char*,PetscInt);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFFooter();
PETSC_INTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif

