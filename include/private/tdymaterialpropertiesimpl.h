#if !defined(TDYMATERIALPROPERTIES_H)
#define TDYMATERIALPROPERTIES_H

#include <petsc.h>

typedef struct _MaterialProp *MaterialProp;

struct _MaterialProp {
    PetscReal *K, *K0;
    PetscReal *porosity;
    PetscReal *Kappa, *Kappa0;
    PetscReal *Cr;
    PetscReal *rhor;
};

PETSC_INTERN PetscErrorCode MaterialPropertiesCreate(PetscInt,PetscInt,MaterialProp*);
PETSC_INTERN void TDySoilDensityFunctionDefault(PetscReal*,PetscReal*);
PETSC_INTERN void TDySpecificSoilHeatFunctionDefault(PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode TDyPermeabilityFunctionDefault(TDy,double*,double*,void*);
PETSC_INTERN PetscErrorCode TDyThermalConductivityFunctionDefault(TDy,double*,double*,void*);
PETSC_INTERN PetscErrorCode TDyPorosityFunctionDefault(TDy,double*,double*,void*);

#endif

