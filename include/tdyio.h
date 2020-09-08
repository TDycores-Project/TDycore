#if !defined(TDYIO_H)
#define TDYIO_H

#include <petsc.h>

typedef enum{
  PetscViewer_Format=0,
  Exodus_Format
} TDyIOFormat;

typedef struct TDyIO *TDyIO;

struct TDyIO {
  PetscBool io_process;
  PetscBool print_intermediate;
  char *exodus_filename;
  char *zonalVarNames[1]; //change
  int num_vars;
  PetscBool exodus_initialized;
  TDyIOFormat format;
};

PETSC_EXTERN PetscErrorCode TDyIOCreate(TDyIO*);
PETSC_EXTERN PetscErrorCode TDyIOSetMode(TDyIO,TDyIOFormat);
PETSC_EXTERN PetscErrorCode TDyIOWriteVec(TDyIO,Vec,const char*,DM,int,PetscReal);
PETSC_EXTERN PetscErrorCode TdyIOInitializeExodus(char*, char*, DM , int);
PETSC_EXTERN PetscErrorCode TdyIOAddExodusTime(char*, PetscReal);
  PETSC_EXTERN PetscErrorCode TdyIOWriteExodusVar(char*, Vec);
PETSC_EXTERN PetscErrorCode TDyIOPrintVec(Vec,const char*,int);
PETSC_EXTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif
