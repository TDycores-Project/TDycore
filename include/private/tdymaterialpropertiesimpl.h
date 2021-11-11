#if !defined(TDYMATERIALPROPERTIES_H)
#define TDYMATERIALPROPERTIES_H

#include <petsc.h>

/// This type represents a set of material properties. It contains only
//⁄ functions that evaluate these properties, and no data (since the storage
/// needed for data is specific to implementations).
typedef struct {

  /// Spatial dimension.
  PetscInt dim;

  /// Contexts provided to material property functions.
  void *porosity_context;
  void *permeability_context;
  void *thermal_conductivity_context;
  void *residual_saturation_context;
  void *soil_density_context;
  void *soil_specific_heat_context;

  /// Material property functions.
  PetscErrorCode (*compute_porosity)(void*,PetscInt,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_permeability)(void*,PetscInt,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_thermal_conductivity)(void*,PetscInt,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_residual_saturation)(void*,PetscInt,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_soil_density)(void*,PetscInt,PetscReal*,PetscReal*);
  PetscErrorCode (*compute_soil_specific_heat)(void*,PetscInt,PetscReal*,PetscReal*);

  /// Material property context destructors.
  void (*porosity_dtor)(void*);
  void (*permeability_dtor)(void*);
  void (*thermal_conductivity_dtor)(void*);
  void (*residual_saturation_dtor)(void*);
  void (*soil_density_dtor)(void*);
  void (*soil_specific_heat_dtor)(void*);

} MaterialProp;

// material model creation/destruction
PETSC_INTERN PetscErrorCode MaterialPropCreate(PetscInt,MaterialProp**);
PETSC_INTERN PetscErrorCode MaterialPropDestroy(MaterialProp*);

// material model setup functions
PETSC_INTERN PetscErrorCode MaterialPropSetPorosity(MaterialProp*, void*, PetscErrorCode (*)(void*,PetscInt,PetscReal*,PetscReal*),void(*)(void*));
PETSC_INTERN PetscErrorCode MaterialPropSetPermeability(MaterialProp*, void*, PetscErrorCode (*)(void*,PetscInt,PetscReal*,PetscReal*),void(*)(void*));
PETSC_INTERN PetscErrorCode MaterialPropSetThermalConductivity(MaterialProp*, void*, PetscErrorCode (*)(void*,PetscInt,PetscReal*,PetscReal*),void(*)(void*));
PETSC_INTERN PetscErrorCode MaterialPropSetResidualSaturation(MaterialProp*, void*, PetscErrorCode (*)(void*,PetscInt,PetscReal*,PetscReal*),void(*)(void*));
PETSC_INTERN PetscErrorCode MaterialPropSetSoilDensity(MaterialProp*, void*, PetscErrorCode (*)(void*,PetscInt,PetscReal*,PetscReal*),void(*)(void*));
PETSC_INTERN PetscErrorCode MaterialPropSetSoilSpecificHeat(MaterialProp*, void*, PetscErrorCode (*)(void*,PetscInt,PetscReal*,PetscReal*),void(*)(void*));

// material model query functions
PETSC_INTERN PetscBool MaterialPropHasPorosity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasPermeability(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasThermalConductivity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasResidualSaturation(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSoilDensity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSoilSpecificHeat(MaterialProp*);

// material quantity computation
PETSC_INTERN PetscErrorCode MaterialPropComputePorosity(MaterialProp*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputePermeability(MaterialProp*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeThermalConductivity(MaterialProp*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeResidualSaturation(MaterialProp*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeSoilDensity(MaterialProp*,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeSoilSpecificHeat(MaterialProp*,PetscInt,PetscReal*,PetscReal*);

// convenience functions
PETSC_INTERN PetscErrorCode MaterialPropSetConstantPorosity(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousPorosity(MaterialProp*, TDyScalarSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantIsotropicPermeability(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousIsotropicPermeability(MaterialProp*, TDyScalarSpatialFunction);
PETSC_INTERN PetscErrorCode MaterialPropSetConstantDiagonalPermeability(MaterialProp*, PetscReal[3]);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousDiagonalPermeability(MaterialProp*, TDyVectorSpatialFunction);
PETSC_INTERN PetscErrorCode MaterialPropSetConstantTensorPermeability(MaterialProp*, PetscReal[9]);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousTensorPermeability(MaterialProp*, TDyTensorSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantIsotropicThermalConductivity(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousIsotropicThermalConductivity(MaterialProp*, TDyScalarSpatialFunction);
PETSC_INTERN PetscErrorCode MaterialPropSetConstantDiagonalThermalConductivity(MaterialProp*, PetscReal[3]);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousDiagonalThermalConductivity(MaterialProp*, TDyVectorSpatialFunction);
PETSC_INTERN PetscErrorCode MaterialPropSetConstantTensorThermalConductivity(MaterialProp*, PetscReal[9]);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousTensorThermalConductivity(MaterialProp*, TDyTensorSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantResidualSaturation(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousResidualSaturation(MaterialProp*, TDyScalarSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantSoilDensity(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousSoilDensity(MaterialProp*, TDyScalarSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantSoilSpecificHeat(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousSoilSpecificHeat(MaterialProp*, TDyScalarSpatialFunction);

#endif

