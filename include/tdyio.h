#if !defined(TDYIO_H)
#define TDYIO_H

#include <petsc.h>

typedef struct IO *IO;

struct IO {
  PetscBool io_process;
  PetscBool print_intermediate;
};

PETSC_EXTERN PetscErrorCode TDyIOCreate(IO*);
PETSC_EXTERN PetscErrorCode TDyIOPrintVec(Vec,char*,int);
PETSC_EXTERN PetscErrorCode TDyIODestroy(IO*);

#endif
