#if !defined(TDYPOROSITY_H)
#define TDYPOROSITY_H

#include <petsc.h>
#include <tdycore.h>

PETSC_EXTERN PetscErrorCode TDyPorosityFunctionDefault(TDy,double*,double*,void*);

#endif
