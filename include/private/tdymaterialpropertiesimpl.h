#if !defined(TDYMATERIALPROPERTIES_H)
#define TDYMATERIALPROPERTIES_H

#include <petsc.h>

typedef struct {
    PetscReal *K, *K0;
    PetscReal *porosity;
    PetscReal *Kappa, *Kappa0;
    PetscReal *Cr;
    PetscReal *rhosoil;

    PetscBool permeability_is_set;
    PetscBool porosity_is_set;
    PetscBool thermal_conductivity_is_set;
    PetscBool soil_specific_heat_is_set;
    PetscBool soil_density_is_set;

} MaterialProp;

PETSC_INTERN PetscErrorCode MaterialPropertiesCreate(PetscInt,PetscInt,MaterialProp**);
PETSC_INTERN PetscErrorCode MaterialPropertiesDestroy(MaterialProp*);
PETSC_INTERN PetscBool TDyIsPermeabilitySet(TDy);
PETSC_INTERN PetscBool TDyIsPorositySet(TDy);
PETSC_INTERN PetscBool TDyIsThermalConductivytSet(TDy);
PETSC_INTERN PetscBool TDyIsSoilSpecificHeatSet(TDy);
PETSC_INTERN PetscBool TDyIsSoilDensitySet(TDy);

PETSC_INTERN PetscErrorCode TDySoilDensityFunctionDefault(TDy,PetscReal*,PetscReal*,void*);
PETSC_INTERN PetscErrorCode TDySoilSpecificHeatFunctionDefault(TDy,PetscReal*,PetscReal*,void*);
PETSC_INTERN PetscErrorCode TDyPermeabilityFunctionDefault(TDy,double*,double*,void*);
PETSC_INTERN PetscErrorCode TDyThermalConductivityFunctionDefault(TDy,double*,double*,void*);
PETSC_INTERN PetscErrorCode TDyPorosityFunctionDefault(TDy,double*,double*,void*);

#endif

