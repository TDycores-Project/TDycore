#if !defined(TDYMATERIALPROPERTIES_H)
#define TDYMATERIALPROPERTIES_H

#include <petsc.h>

PETSC_INTERN void TDySoilDensityFunctionDefault(PetscReal*,PetscReal*);
PETSC_INTERN void TDySpecificSoilHeatFunctionDefault(PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode TDyPermeabilityFunctionDefault(TDy,double*,double*,void*);
PETSC_INTERN PetscErrorCode TDyThermalConductivityFunctionDefault(TDy,double*,double*,void*);
PETSC_INTERN PetscErrorCode TDyPorosityFunctionDefault(TDy,double*,double*,void*);

#endif

