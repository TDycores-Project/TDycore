#if !defined(TDYIO_H)
#define TDYIO_H

#include <petsc.h>

typedef enum{
  PetscViewerASCIIFormat=0,
  ExodusFormat,
  HDF5Format,
  NullFormat
} TDyIOFormat;

typedef struct TDyIO *TDyIO;

struct TDyIO {
  PetscBool io_process;
  PetscBool print_intermediate;
  char filename[PETSC_MAX_PATH_LEN];
  char zonalVarNames[1][PETSC_MAX_PATH_LEN];
  int num_vars;
  TDyIOFormat format;
  int num_times;
};

PETSC_EXTERN PetscErrorCode TDyIOCreate(TDyIO*);
PETSC_EXTERN PetscErrorCode TDyIOSetMode(TDy,TDyIOFormat);
PETSC_EXTERN PetscErrorCode TDyIOWriteVec(TDy);
PETSC_EXTERN PetscErrorCode TdyIOInitializeExodus(char*,char*[],DM ,int);
PETSC_EXTERN PetscErrorCode TdyIOInitializeHDF5(char*,DM);
PETSC_EXTERN PetscErrorCode TdyIOAddExodusTime(char*,PetscReal,TDyIO);
PETSC_EXTERN PetscErrorCode TdyIOWriteExodusVar(char*,Vec,TDyIO);
PETSC_EXTERN PetscErrorCode TdyIOWriteHDF5Var(char*,Vec,PetscReal);
PETSC_EXTERN PetscErrorCode TDyIOWriteAsciiViewer(Vec,PetscReal);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFHeader(PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFAttribute(char*,PetscInt);
PETSC_EXTERN PetscErrorCode TDyIOWriteXMFFooter();
PETSC_EXTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif

