#if !defined(TDYMATERIALPROPERTIES_H)
#define TDYMATERIALPROPERTIES_H

#include <petsc.h>

/// This is a vectorized function pointer type used to compute all material
/// properties. It allows the specification of a space-time-dependent function
/// that can return scalars and tensors.
typedef PetscErrorCode (*MaterialPropFunc)(void*,PetscReal,PetscInt,PetscReal*,PetscReal*);

// This is a destructor function pointer type for material properties.
typedef PetscErrorCode (*MaterialPropDtor)(void*);

/// This type represents a set of material properties. It contains only
/// functions that evaluate these properties, and no data (since the storage
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
  void *saline_diffusivity_context;
  void *saline_molecular_weight_context;

  /// Material property functions.
  MaterialPropFunc compute_porosity;
  MaterialPropFunc compute_permeability;
  MaterialPropFunc compute_thermal_conductivity;
  MaterialPropFunc compute_residual_saturation;
  MaterialPropFunc compute_soil_density;
  MaterialPropFunc compute_soil_specific_heat;
  MaterialPropFunc compute_saline_diffusivity;
  MaterialPropFunc compute_saline_molecular_weight;

  /// Material property context destructors.
  MaterialPropDtor porosity_dtor;
  MaterialPropDtor permeability_dtor;
  MaterialPropDtor thermal_conductivity_dtor;
  MaterialPropDtor residual_saturation_dtor;
  MaterialPropDtor soil_density_dtor;
  MaterialPropDtor soil_specific_heat_dtor;
  MaterialPropDtor saline_diffusivity_dtor;
  MaterialPropDtor saline_molecular_weight_dtor;

} MaterialProp;

// material model creation/destruction
PETSC_INTERN PetscErrorCode MaterialPropCreate(PetscInt,MaterialProp**);
PETSC_INTERN PetscErrorCode MaterialPropDestroy(MaterialProp*);

// material model setup functions
PETSC_INTERN PetscErrorCode MaterialPropSetPorosity(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);
PETSC_INTERN PetscErrorCode MaterialPropSetPermeability(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);
PETSC_INTERN PetscErrorCode MaterialPropSetThermalConductivity(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);
PETSC_INTERN PetscErrorCode MaterialPropSetResidualSaturation(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);
PETSC_INTERN PetscErrorCode MaterialPropSetSoilDensity(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);
PETSC_INTERN PetscErrorCode MaterialPropSetSoilSpecificHeat(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);
PETSC_INTERN PetscErrorCode MaterialPropSetSalineDiffusivity(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);
PETSC_INTERN PetscErrorCode MaterialPropSetSalineMolecularWeight(MaterialProp*, void*, MaterialPropFunc, MaterialPropDtor);

// material model query functions
PETSC_INTERN PetscBool MaterialPropHasPorosity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasPermeability(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasThermalConductivity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasResidualSaturation(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSoilDensity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSoilSpecificHeat(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSalineDiffusivity(MaterialProp*);
PETSC_INTERN PetscBool MaterialPropHasSalineMolecularWeight(MaterialProp*);

// material quantity computation
PETSC_INTERN PetscErrorCode MaterialPropComputePorosity(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputePermeability(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeThermalConductivity(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeResidualSaturation(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeSoilDensity(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeSoilSpecificHeat(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeSalineDiffusivity(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode MaterialPropComputeSalineMolecularWeight(MaterialProp*,PetscReal,PetscInt,PetscReal*,PetscReal*);

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

PETSC_INTERN PetscErrorCode MaterialPropSetConstantIsotropicSalineDiffusivity(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousIsotropicSalineDiffusivity(MaterialProp*, TDyScalarSpatialFunction);
PETSC_INTERN PetscErrorCode MaterialPropSetConstantDiagonalSalineDiffusivity(MaterialProp*, PetscReal[3]);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousDiagonalSalineDiffusivity(MaterialProp*, TDyVectorSpatialFunction);
PETSC_INTERN PetscErrorCode MaterialPropSetConstantTensorSalineDiffusivity(MaterialProp*, PetscReal[9]);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousTensorSalineDiffusivity(MaterialProp*, TDyTensorSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantResidualSaturation(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousResidualSaturation(MaterialProp*, TDyScalarSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantSoilDensity(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousSoilDensity(MaterialProp*, TDyScalarSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantSoilSpecificHeat(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousSoilSpecificHeat(MaterialProp*, TDyScalarSpatialFunction);

PETSC_INTERN PetscErrorCode MaterialPropSetConstantSalineMolecularWeight(MaterialProp*, PetscReal);
PETSC_INTERN PetscErrorCode MaterialPropSetHeterogeneousSalineMolecularWeight(MaterialProp*, TDyTensorSpatialFunction);
#endif

