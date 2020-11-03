#if !defined(TDYIO_H)
#define TDYIO_H

#include <petsc.h>

typedef enum{
  PetscViewerASCIIFormat=0,
  ExodusFormat
} TDyIOFormat;

typedef struct TDyIO *TDyIO;

struct TDyIO {
  PetscBool io_process;
  PetscBool print_intermediate;
  char exodus_filename[PETSC_MAX_PATH_LEN];
  char zonalVarNames[1][PETSC_MAX_PATH_LEN];
  int num_vars;
  TDyIOFormat format;
  int num_times;
};

PETSC_INTERN PetscErrorCode TDyIOCreate(TDyIO*);
PETSC_EXTERN PetscErrorCode TDyIOSetMode(TDyIO,TDyIOFormat);
PETSC_EXTERN PetscErrorCode TDyIOWriteVec(TDy);
PETSC_INTERN PetscErrorCode TdyIOInitializeExodus(char*,char*[],DM ,int);
PETSC_INTERN PetscErrorCode TdyIOAddExodusTime(char*,PetscReal,TDyIO);
PETSC_INTERN PetscErrorCode TdyIOWriteExodusVar(char*,Vec,TDyIO);
PETSC_INTERN PetscErrorCode TDyIOPrintVec(Vec,PetscReal);
PETSC_INTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif
