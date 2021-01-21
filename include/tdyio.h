#if !defined(TDYIO_H)
#define TDYIO_H

#include <petsc.h>

typedef enum{
  PetscViewerASCIIFormat=0,
  ExodusFormat,
  HDF5Format,
  NullFormat
} TDyIOFormat;

typedef struct _p_TDyIO* TDyIO;

PETSC_EXTERN PetscErrorCode TDyIOSetIOProcess(TDyIO,PetscBool);
PETSC_EXTERN PetscErrorCode TDyIOSetPrintIntermediate(TDyIO,PetscBool);
PETSC_EXTERN PetscErrorCode TDyIOSetMode(TDy,TDyIOFormat);
PETSC_EXTERN PetscErrorCode TDyIOWriteVec(TDy);

#endif

