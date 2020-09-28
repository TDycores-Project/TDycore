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
  char *exodus_filename;
  char *zonalVarNames[1]; 
  int num_vars;
  TDyIOFormat format;
  int num_times;
};

PETSC_EXTERN PetscErrorCode TDyIOCreate(TDyIO*);
PETSC_EXTERN PetscErrorCode TDyIOSetMode(TDyIO,TDyIOFormat);
PETSC_EXTERN PetscErrorCode TDyIOWriteVec(TDy);
PETSC_EXTERN PetscErrorCode TdyIOInitializeExodus(char*,char*[],DM ,int);
PETSC_EXTERN PetscErrorCode TdyIOAddExodusTime(char*,PetscReal,TDyIO);
PETSC_EXTERN PetscErrorCode TdyIOWriteExodusVar(char*,Vec,TDyIO);
PETSC_EXTERN PetscErrorCode TDyIOPrintVec(Vec,PetscReal);
PETSC_EXTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif
