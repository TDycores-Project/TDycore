#if !defined(TDYOPTIONS_H)
#define TDYOPTIONS_H

#include <petsc.h>

typedef struct{
    PetscReal gravity_constant;

    // Default values for material properties
    PetscReal default_porosity;
    PetscReal default_permeability;
    PetscReal default_soil_density;
    PetscReal default_soil_specific_heat;
    PetscReal default_thermal_conductivity;

    // Default values for characteristic curves
    PetscReal default_residual_saturation;
    PetscReal default_gardner_n;
    PetscReal default_vangenuchten_m;
    PetscReal default_vangenutchen_alpha;

} TDyOptions;

#endif