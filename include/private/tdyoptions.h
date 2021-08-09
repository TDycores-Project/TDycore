#if !defined(TDYOPTIONS_H)
#define TDYOPTIONS_H

#include <petsc.h>

typedef struct{
    PetscReal gravity_constant;

    // Material properties
    PetscReal porosity;
    PetscReal permeability;
    PetscReal soil_density;
    PetscReal soil_specific_heat;
    PetscReal thermal_conductivity;

    // Characteristic curves
    PetscReal residual_saturation;
    PetscReal gardner_n;
    PetscReal vangenuchten_m;
    PetscReal vangenuchten_alpha;

} TDyOptions;

#endif
