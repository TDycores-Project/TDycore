#if !defined(RICHARDS_H)
#define RICHARDS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include <tdycore.h>
#include <private/tdycoreimpl.h>

typedef struct Richards *Richards;

struct Richards {
  TDy tdy;
  Vec U;
  SNES snes;

  PetscBool io_process;
  PetscBool print_intermediate;

  PetscScalar time;
  PetscScalar dtime;
  PetscScalar final_time;
  PetscInt istep;
};

PETSC_EXTERN PetscErrorCode RichardsCreate(Richards*);
PETSC_EXTERN PetscErrorCode RichardsInitialize(Richards);
PETSC_EXTERN PetscErrorCode RichardsPrintVec(Vec,char*,int);
PETSC_EXTERN PetscErrorCode RichardsRunToTime(Richards,PetscReal);
PETSC_EXTERN PetscErrorCode RichardsDestroy(Richards*);

#endif
