#if !defined(RICHARDS_H)
#define RICHARDS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include <tdycore.h>

typedef struct Richards Richards;

struct Richards {
  TDy tdy;
  Vec U;
  Vec F;
  Mat J;
};

PETSC_EXTERN PetscErrorCode RichardsCreate(Richards **);
PETSC_EXTERN PetscErrorCode RichardsRunToTime(Richards,PetscReal);
PETSC_EXTERN PetscErrorCode RichardsDestroy(Richards **);

#endif
