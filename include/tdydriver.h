#if !defined(TDYDRIVER_H)
#define TDYDRIVER_H

#include <petsc.h>
#include <private/tdycoreimpl.h>
#include <tdycore.h>
#include <tdypermeability.h>
#include <tdyporosity.h>
#include <tdyrichards.h>

PETSC_EXTERN PetscErrorCode TDyDriverInitializeTDy(TDy tdy);

#endif
