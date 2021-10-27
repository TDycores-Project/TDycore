#if !defined(TDYMATERIALPROPERTIES_H)
#define TDYMATERIALPROPERTIES_H

#include <petsc.h>

typedef struct {
  // Spatial dimension (for tensor properties)
  int dim;

  // Number of points at which data is stored
  int num_points;

  // Contexts provided to material property functions.
  void *porosity_context;
  void *permeability_context;
  void *thermal_conductivity_context;
  void *residual_saturation_context;
  void *soil_density_context;
  void *soil_specific_heat_context;

  // Material property functions.
  PetscErrorCode (*compute_porosity)(void*,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_permeability)(void*,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_thermal_conductivity)(void*,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_residual_saturation)(void*,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_soil_density)(void*,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_soil_specific_heat)(void*, PetscReal*,PetscReal*);

  // Material data.
  PetscReal *K, *K0;
  PetscReal *porosity;
  PetscReal *Kappa, *Kappa0;
  PetscReal *Cr;
  PetscReal *rhosoil;

} MaterialProp;

PETSC_INTERN PetscErrorCode MaterialPropCreate(void*,PetscInt,PetscInt,MaterialProp**);
PETSC_INTERN PetscErrorCode MaterialPropDestroy(MaterialProp*);
PETSC_INTERN PetscErrorCode MaterialPropSetPorosity(MaterialProp*, void*, PetscErrorCode (*)(void*,int,PetscReal*,PetscReal*));
PETSC_INTERN PetscErrorCode MaterialPropSetPermeability(MaterialProp*, void*, PetscErrorCode (*)(void*,int,PetscReal*,PetscReal*));
PETSC_INTERN PetscErrorCode MaterialPropSetThermalConductivity(MaterialProp*, void*, PetscErrorCode (*)(void*,int,PetscReal*,PetscReal*));
PETSC_INTERN PetscErrorCode MaterialPropSetResidualSaturation(MaterialProp*, void*, PetscErrorCode (*)(void*,int,PetscReal*,PetscReal*));
PETSC_INTERN PetscErrorCode MaterialPropSetSoilDensity(MaterialProp*, void*, PetscErrorCode (*)(void*,int,PetscReal*,PetscReal*));
PETSC_INTERN PetscErrorCode MaterialPropSetSoilSpecificHeat(MaterialProp*, void*, PetscErrorCode (*)(void*,int,PetscReal*,PetscReal*));
PETSC_INTERN PetscBool MaterialPropHasPorosity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasPermeability(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasThermalConductivity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasResidualSaturation(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSoilDensity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSoilSpecificHeat(MaterialProp*);
PETSC_INTERN PetscErrorCode MaterialPropUpdate(MaterialProp*);

#endif

