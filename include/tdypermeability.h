#if !defined(TDYPERMEABILITY_H)
#define TDYPERMEABILITY_H

#include <petsc.h>
#include <tdycore.h>

PETSC_INTERN PetscErrorCode TDyPermeabilityFunctionDefault(TDy,double*,double*,void*);
PETSC_INTERN PetscErrorCode TDyThermalConductivityFunctionDefault(TDy,double*,double*,void*);

#endif
