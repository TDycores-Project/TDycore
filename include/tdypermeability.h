#if !defined(TDYPERMEABILITY_H)
#define TDYPERMEABILITY_H

#include <petsc.h>
#include <tdycore.h>

PETSC_EXTERN PetscErrorCode TDyPermeabilityFunctionDefault(TDy,double*,double*,void*);
PETSC_EXTERN PetscErrorCode TDyThermalConductivityFunctionDefault(TDy,double*,double*,void*);

#endif
