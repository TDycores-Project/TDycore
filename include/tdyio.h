#if !defined(TDYIO_H)
#define TDYIO_H

#include <petsc.h>

typedef struct TDyIO *TDyIO;

struct TDyIO {
  PetscBool io_process;
  PetscBool print_intermediate;
};

PETSC_EXTERN PetscErrorCode TDyIOCreate(TDyIO*);
PETSC_EXTERN PetscErrorCode TDyIOPrintVec(Vec,const char*,int);
PETSC_EXTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif
