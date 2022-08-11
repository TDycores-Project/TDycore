#include <private/tdycoreimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <private/tdymemoryimpl.h>
#include <tdytimers.h>
#include <private/tdyoptions.h>

/// Creates a new MaterialProp instance without any functions assigned for
/// computing its properties.
/// @param [in] dim The spatial dimension of the domain associated with the
///                 material.
/// @param [out] matprop Storage for the new instance
PetscErrorCode MaterialPropCreate(PetscInt dim, MaterialProp **matprop) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = TDyAlloc(sizeof(MaterialProp), matprop); CHKERRQ(ierr);
  (*matprop)->dim = dim;

  PetscFunctionReturn(0);
}

/// Destroys the given instance (and any context pointers managed by it).
/// @param [inout] matprop A MaterialProp instance
PetscErrorCode MaterialPropDestroy(MaterialProp *matprop) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MaterialPropSetPorosity(matprop, NULL, NULL, NULL);
  MaterialPropSetPermeability(matprop, NULL, NULL, NULL);
  MaterialPropSetThermalConductivity(matprop, NULL, NULL, NULL);
  MaterialPropSetResidualSaturation(matprop, NULL, NULL, NULL);
  MaterialPropSetSoilDensity(matprop, NULL, NULL, NULL);
  MaterialPropSetSoilSpecificHeat(matprop, NULL, NULL, NULL);
  MaterialPropSetSalineDiffusivity(matprop, NULL, NULL, NULL);
  MaterialPropSetSalineMolecularWeight(matprop, NULL, NULL, NULL);
  ierr = TDyFree(matprop); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the permeability.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the permeability at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetPermeability(MaterialProp *matprop, void *context,
                                           PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                           PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->permeability_context && matprop->permeability_dtor)
    matprop->permeability_dtor(matprop->permeability_context);
  matprop->permeability_context = context;
  matprop->compute_permeability = f;
  matprop->permeability_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the porosity.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the porosity at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetPorosity(MaterialProp *matprop, void *context,
                                       PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                       PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->porosity_context && matprop->porosity_dtor)
    matprop->porosity_dtor(matprop->porosity_context);
  matprop->porosity_context = context;
  matprop->compute_porosity = f;
  matprop->porosity_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the thermal conductivity.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the thermal conductivity at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetThermalConductivity(MaterialProp *matprop, void *context,
                                                  PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                                  PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->thermal_conductivity_context && matprop->thermal_conductivity_dtor)
    matprop->thermal_conductivity_dtor(matprop->thermal_conductivity_context);
  matprop->thermal_conductivity_context = context;
  matprop->compute_thermal_conductivity = f;
  matprop->thermal_conductivity_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the residual saturation.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the residual saturation at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetResidualSaturation(MaterialProp *matprop, void *context,
                                                 PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                                 PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->residual_saturation_context && matprop->residual_saturation_dtor)
    matprop->residual_saturation_dtor(matprop->residual_saturation_context);
  matprop->residual_saturation_context = context;
  matprop->compute_residual_saturation = f;
  matprop->residual_saturation_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the soil density.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the soil density at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetSoilDensity(MaterialProp *matprop, void *context,
                                          PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                          PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->soil_density_context && matprop->soil_density_dtor)
    matprop->soil_density_dtor(matprop->soil_density_context);
  matprop->soil_density_context = context;
  matprop->compute_soil_density = f;
  matprop->soil_density_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the soil specific heat.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the soil density at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetSoilSpecificHeat(MaterialProp *matprop, void *context,
                                               PetscErrorCode(*f)(void*,int,PetscReal*,PetscReal*),
                                               PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->soil_specific_heat_context && matprop->soil_specific_heat_dtor)
    matprop->soil_specific_heat_dtor(matprop->soil_specific_heat_context);
  matprop->soil_specific_heat_context = context;
  matprop->compute_soil_specific_heat = f;
  matprop->soil_specific_heat_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the saline diffusivity.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the saline diffusivity at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetSalineDiffusivity(MaterialProp *matprop, void *context,
                                                PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                                PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->saline_diffusivity_context && matprop->saline_diffusivity_dtor)
    matprop->saline_diffusivity_dtor(matprop->saline_diffusivity_context);
  matprop->saline_diffusivity_context = context;
  matprop->compute_saline_diffusivity = f;
  matprop->saline_diffusivity_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Sets the function used to compute the saline molecular weight.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the saline molecular weight at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetSalineMolecularWeight(MaterialProp *matprop, void *context,
                                                    PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                                    PetscErrorCode (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->saline_molecular_weight_context && matprop->saline_molecular_weight_dtor)
    matprop->saline_molecular_weight_dtor(matprop->saline_molecular_weight_context);
  matprop->saline_molecular_weight_context = context;
  matprop->compute_saline_molecular_weight = f;
  matprop->saline_molecular_weight_dtor = dtor;
  PetscFunctionReturn(0);
}

/// Returns true if this instance can compute porosities, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasPorosity(MaterialProp* matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_porosity != NULL);
}

/// Returns true if this instance can compute permeabilities, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasPermeability(MaterialProp *matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_permeability != NULL);
}

/// Returns true if this instance can compute thermal conductivities, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasThermalConductivity(MaterialProp *matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_thermal_conductivity != NULL);
}

/// Returns true if this instance can compute residual saturations, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasResidualSaturation(MaterialProp *matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_residual_saturation != NULL);
}

/// Returns true if this instance can compute soil densities, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasSoilDensity(MaterialProp *matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_soil_density != NULL);
}

/// Returns true if this instance can compute soil specific heats, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasSoilSpecificHeat(MaterialProp *matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_soil_specific_heat != NULL);
}

/// Returns true if this instance can compute saline diffusivities, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasSalineDiffusivity(MaterialProp *matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_saline_diffusivity != NULL);
}

/// Returns true if this instance can compute saline molecular weights, false
/// otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasSalineMolecularWeight(MaterialProp *matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->compute_saline_molecular_weight != NULL);
}

/// Computes the porosity at the given set of points (if the material properties
/// have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] porosity The values of the porosity at each point
PetscErrorCode MaterialPropComputePorosity(MaterialProp *matprop,
                                           PetscInt n, PetscReal *x,
                                           PetscReal *porosity) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_porosity) {
    ierr = matprop->compute_porosity(matprop->porosity_context,
                                     n, x, porosity); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the permeability tensor at the given set of points (if the material
/// properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] permeability The values of the permeability at each point
PetscErrorCode MaterialPropComputePermeability(MaterialProp *matprop,
                                               PetscInt n, PetscReal *x,
                                               PetscReal* permeability) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_permeability) {
    ierr = matprop->compute_permeability(matprop->permeability_context,
                                         n, x, permeability); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the thermal conductivity tensor at the given set of points (if the
/// material properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] conductivity The values of the conductivity at each point
PetscErrorCode MaterialPropComputeThermalConductivity(MaterialProp *matprop,
                                                      PetscInt n, PetscReal *x,
                                                      PetscReal *conductivity) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_thermal_conductivity) {
    ierr = matprop->compute_thermal_conductivity(matprop->thermal_conductivity_context,
                                                 n, x, conductivity); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the residual saturation at the given set of points (if the material
/// properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] saturation The values of the residual saturation at each point
PetscErrorCode MaterialPropComputeResidualSaturation(MaterialProp *matprop,
                                                     PetscInt n, PetscReal *x,
                                                     PetscReal *saturation) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_residual_saturation) {
    ierr = matprop->compute_residual_saturation(matprop->residual_saturation_context,
                                                n, x, saturation); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the soil density at the given set of points (if the material
/// properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] density The values of the density at each point
PetscErrorCode MaterialPropComputeSoilDensity(MaterialProp *matprop,
                                              PetscInt n, PetscReal *x,
                                              PetscReal *density) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_soil_density) {
    ierr = matprop->compute_soil_density(matprop->soil_density_context,
                                         n, x, density); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the soil specific heat at the given set of points (if the material
/// properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] specific_heat The values of the specific heat at each point
PetscErrorCode MaterialPropComputeSoilSpecificHeat(MaterialProp *matprop,
                                                   PetscInt n, PetscReal *x,
                                                   PetscReal *specific_heat) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_soil_specific_heat) {
    ierr = matprop->compute_soil_specific_heat(matprop->soil_specific_heat_context,
                                               n, x, specific_heat); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the saline diffusivity tensor at the given set of points (if the
/// material properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] conductivity The values of the diffusivity at each point
PetscErrorCode MaterialPropComputeSalineDiffusivity(MaterialProp *matprop,
                                                    PetscInt n, PetscReal *x,
                                                    PetscReal *diffusivity) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_saline_diffusivity) {
    ierr = matprop->compute_saline_diffusivity(matprop->saline_diffusivity_context,
                                               n, x, diffusivity); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the saline molecular weight at the given set of points (if the
/// material properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] conductivity The values of the molecular weight at each point
PetscErrorCode MaterialPropComputeSalineMolecularWeight(MaterialProp *matprop,
                                                        PetscInt n, PetscReal *x,
                                                        PetscReal *weight) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_saline_molecular_weight) {
    ierr = matprop->compute_saline_molecular_weight(matprop->saline_molecular_weight_context,
                                                    n, x, weight); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// Generic wrapper struct that serves as a context for functions in this file.
typedef struct WrapperStruct {
  // Spatial dimension
  PetscInt dim;
  // Constant value (if relevant--need 9 values to store a full tensor in 3D)
  PetscReal value[9];
  // ScalarFunctions, VectorFunctions, and TensorFunctions can all be stored
  // here.
  void (*func)(PetscInt, PetscReal*, PetscReal*);
} WrapperStruct;

// Generic constructor for contexts for constant functions in this file.
static PetscErrorCode CreateConstantContext(PetscInt dim, PetscReal value[9],
                                            void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper;
  ierr = TDyAlloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  wrapper->dim = dim;
  memcpy(wrapper->value, value, 9*sizeof(PetscReal));
  wrapper->func = NULL;
  *context = wrapper;
  PetscFunctionReturn(0);
}

// Generic constructor for contexts for spatial functions in this file.
static PetscErrorCode CreateSpatialContext(PetscInt dim,
                                           void (*func)(PetscInt, PetscReal*, PetscReal*),
                                           void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper;
  ierr = TDyAlloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  wrapper->dim = dim;
  memset(wrapper->value, 0, 9*sizeof(PetscReal));
  wrapper->func = func;
  *context = wrapper;
  PetscFunctionReturn(0);
}

// Generic function to call for constant functions that compute scalar
// quantities in this file.
static PetscErrorCode ConstantScalarWrapperFunction(void *context, PetscInt n,
                                                    PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  for (PetscInt i = 0; i < n; ++i) {
    f[i] = wrapper->value[0];
  }
  PetscFunctionReturn(0);
}

// Generic function to call for scalar spatial functions that compute scalar
// quantities in this file.
static PetscErrorCode ScalarWrapperFunction(void *context, PetscInt n,
                                            PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  wrapper->func(n, x, f);
  PetscFunctionReturn(0);
}

// Generic function to call for scalar constant functions that compute isotropic
// tensor quantities in this file.
static PetscErrorCode ConstantIsotropicTensorWrapperFunction(void *context, PetscInt n,
                                                             PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  PetscInt dim = wrapper->dim;
  PetscInt dim2 = dim*dim;
  for (PetscInt i = 0; i < n; ++i) {
    for(PetscInt j = 0; j < dim; ++j) {
      f[dim2*i+j*dim+j] = wrapper->value[0];
    }
  }
  PetscFunctionReturn(0);
}

// Generic function to call for scalar spatial functions that compute isotropic
// tensor quantities in this file.
static PetscErrorCode IsotropicTensorWrapperFunction(void *context, PetscInt n,
                                                     PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  PetscReal values[n];
  wrapper->func(n, x, values);
  PetscInt dim = wrapper->dim;
  PetscInt dim2 = dim*dim;
  for (PetscInt i = 0; i < n; ++i) {
    for(PetscInt j = 0; j < dim; ++j) {
      f[dim2*i+j*dim+j] = values[i];
    }
  }
  PetscFunctionReturn(0);
}

// Generic function to call for vector constannt functions that compute
// diagonal anisotropic tensor quantities in this file.
static PetscErrorCode ConstantDiagonalTensorWrapperFunction(void *context, PetscInt n,
                                                            PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  PetscInt dim = wrapper->dim;
  PetscInt dim2 = dim*dim;
  for (PetscInt i = 0; i < n; ++i) {
    for(PetscInt j = 0; j < dim; ++j) {
      f[dim2*i+j*dim+j] = wrapper->value[j];
    }
  }
  PetscFunctionReturn(0);
}

// Generic function to call for vector spatial functions that compute
// diagonal anisotropic tensor quantities in this file.
static PetscErrorCode DiagonalTensorWrapperFunction(void *context, PetscInt n,
                                                    PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  PetscInt dim = wrapper->dim;
  PetscInt dim2 = dim*dim;
  PetscReal values[3*n];
  wrapper->func(n, x, values);
  for (PetscInt i = 0; i < n; ++i) {
    for(PetscInt j = 0; j < dim; ++j) {
      f[dim2*i+j*dim+j] = values[3*i+j];
    }
  }
  PetscFunctionReturn(0);
}

// Generic function to call for vector spatial functions that compute
// full anisotropic tensor quantities in this file.
static PetscErrorCode ConstantFullTensorWrapperFunction(void *context, PetscInt n,
                                                        PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  PetscInt dim = wrapper->dim;
  PetscInt dim2 = dim*dim;
  for (PetscInt i = 0; i < n; ++i) {
    for(PetscInt j = 0; j < dim2; ++j) {
      f[dim2*i+j] = wrapper->value[j];
    }
  }
  PetscFunctionReturn(0);
}

// Generic function to call for vector spatial functions that compute
// full anisotropic tensor quantities in this file.
static PetscErrorCode FullTensorWrapperFunction(void *context, PetscInt n,
                                                PetscReal *x, PetscReal *f) {
  PetscFunctionBegin;
  WrapperStruct *wrapper = context;
  wrapper->func(n, x, f);
  PetscFunctionReturn(0);
}

// These macros are used to support the various material model quantities.
// In each case, a function is declared that calls
// MaterialPropSetCapCamelName with a struct populated by an appropriate datum.
// Some examples:
// Quantity name:         CapCamelName:
// porosity               Porosity
// thermal conductivity   ThermalConductivity

// Defines a function named MaterialPropSetConstantCapCamelName that assigns
// the given quantity to a constant scalar.
#define DEFINE_SET_CONSTANT_SCALAR(CapCamelName) \
PetscErrorCode MaterialPropSetConstant##CapCamelName(MaterialProp *matprop, \
                                                     PetscReal scalar) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  PetscReal value[9]; \
  value[0] = scalar; \
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, ConstantScalarWrapperFunction, \
                                 TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

// Defines a function named MaterialPropSetHeterogeneousCapCamelName that
// assigns the given quantity to a scalar spatial function.
#define DEFINE_SET_HETEROGENEOUS_SCALAR(CapCamelName) \
PetscErrorCode MaterialPropSetHeterogeneous##CapCamelName(MaterialProp *matprop, \
                                                          TDyScalarSpatialFunction f) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, ScalarWrapperFunction, \
                                       TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

// Defines a function named MaterialPropSetConstantIsotropicCapCamelName that
// assigns the given quantity to a constant isotropic tensor.
#define DEFINE_SET_CONSTANT_ISOTROPIC_TENSOR(CapCamelName) \
PetscErrorCode MaterialPropSetConstantIsotropic##CapCamelName(MaterialProp *matprop, \
                                                              PetscReal tensor) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  PetscReal value[9]; \
  value[0] = tensor; \
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, ConstantIsotropicTensorWrapperFunction, \
                                      TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

// Defines a function named MaterialPropSetHeterogeneousIsotropicCapCamelName
// that assigns the given quantity to a isotropic tensor spatial function.
#define DEFINE_SET_HETEROGENEOUS_ISOTROPIC_TENSOR(CapCamelName) \
PetscErrorCode MaterialPropSetHeterogeneousIsotropic##CapCamelName(MaterialProp *matprop, \
                                                                   TDyScalarSpatialFunction f) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, IsotropicTensorWrapperFunction, \
                                      TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

// Defines a function named MaterialPropSetConstantDiagonalCapCamelName
// that assigns the given quantity to a constant diagonal tensor.
#define DEFINE_SET_CONSTANT_DIAGONAL_TENSOR(CapCamelName) \
PetscErrorCode MaterialPropSetConstantDiagonal##CapCamelName(MaterialProp *matprop, \
                                                             PetscReal tensor[3]) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  PetscReal value[9]; \
  value[0] = tensor[0]; value[1] = tensor[1]; value[2] = tensor[2]; \
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, ConstantDiagonalTensorWrapperFunction, \
                                       TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

// Defines a function named MaterialPropSetHeterogeneousDiagonalCapCamelName
// that assigns the given quantity to a diagonal tensor spatial function.
#define DEFINE_SET_HETEROGENEOUS_DIAGONAL_TENSOR(CapCamelName) \
PetscErrorCode MaterialPropSetHeterogeneousDiagonal##CapCamelName(MaterialProp *matprop, \
                                                                  TDyVectorSpatialFunction f) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, DiagonalTensorWrapperFunction, \
                                       TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

// Defines a function named MaterialPropSetConstantTensorCapCamelName
// that assigns the given quantity to a constant (full) tensor.
#define DEFINE_SET_CONSTANT_TENSOR(CapCamelName) \
PetscErrorCode MaterialPropSetConstantTensor##CapCamelName(MaterialProp *matprop, \
                                                           PetscReal tensor[9]) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  ierr = CreateConstantContext(matprop->dim, tensor, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, ConstantFullTensorWrapperFunction, \
                                       TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

// Defines a function named MaterialPropSetHeterogeneousTensorCapCamelName
// that assigns the given quantity to a (full) tensor spatial function.
#define DEFINE_SET_HETEROGENEOUS_TENSOR(CapCamelName) \
PetscErrorCode MaterialPropSetHeterogeneousTensor##CapCamelName(MaterialProp *matprop, \
                                                                TDyTensorSpatialFunction f) { \
  PetscErrorCode ierr; \
  PetscFunctionBegin; \
\
  void *context; \
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr); \
  ierr = MaterialPropSet##CapCamelName(matprop, context, FullTensorWrapperFunction, \
                                       TDyFree); CHKERRQ(ierr); \
  PetscFunctionReturn(0); \
}

/// Sets the porosity function to the given constant value.
DEFINE_SET_CONSTANT_SCALAR(Porosity)

/// Sets the porosity function to the given spatial scalar function f.
DEFINE_SET_HETEROGENEOUS_SCALAR(Porosity)

/// Sets the permeability function to a given constant value that provides an
/// isotropic tensor.
DEFINE_SET_CONSTANT_ISOTROPIC_TENSOR(Permeability)

/// Sets the permeability function to the given spatial scalar function f that
/// provides an isotropic tensor.
DEFINE_SET_HETEROGENEOUS_ISOTROPIC_TENSOR(Permeability)

/// Sets the permeability function to a given constant diagonal tensor.
DEFINE_SET_CONSTANT_DIAGONAL_TENSOR(Permeability)

/// Sets the permeability function to a given spatial vector function f that
/// computes a diagonal anisotropic tensor.
DEFINE_SET_HETEROGENEOUS_DIAGONAL_TENSOR(Permeability)

/// Sets the permeability function to a given constant tensor.
DEFINE_SET_CONSTANT_TENSOR(Permeability)

/// Sets the permeability function to a given spatial tensor function f that
/// computes a full anisotropic tensor.
DEFINE_SET_HETEROGENEOUS_TENSOR(Permeability)

/// Sets the thermal conductivity function to a given constant isotropic value.
DEFINE_SET_CONSTANT_ISOTROPIC_TENSOR(ThermalConductivity)

/// Sets the thermal conductivity function to the given spatial scalar function f that
/// provides an isotropic tensor.
DEFINE_SET_HETEROGENEOUS_ISOTROPIC_TENSOR(ThermalConductivity)

/// Sets the thermal conductivity function to a given constant diagonal tensor.
DEFINE_SET_CONSTANT_DIAGONAL_TENSOR(ThermalConductivity)

/// Sets the thermal conductivity function to a given spatial vector function f that
/// computes a diagonal anisotropic tensor.
DEFINE_SET_HETEROGENEOUS_DIAGONAL_TENSOR(ThermalConductivity)

/// Sets the thermal conductivity function to a given constant tensor.
DEFINE_SET_CONSTANT_TENSOR(ThermalConductivity)

/// Sets the thermal conductivity function to a given spatial tensor function f that
/// computes a full anisotropic conductivity tensor.
DEFINE_SET_HETEROGENEOUS_TENSOR(ThermalConductivity)

/// Sets the residual saturation function to the given constant value.
DEFINE_SET_CONSTANT_SCALAR(ResidualSaturation)

/// Sets the residual saturation function to the given spatial scalar function f.
DEFINE_SET_HETEROGENEOUS_SCALAR(ResidualSaturation)

/// Sets the soil density function to the given constant value.
DEFINE_SET_CONSTANT_SCALAR(SoilDensity)

/// Sets the soil density function to the given spatial scalar function f.
DEFINE_SET_HETEROGENEOUS_SCALAR(SoilDensity)

/// Sets the soil specific heat function to the given constant value.
DEFINE_SET_CONSTANT_SCALAR(SoilSpecificHeat)

/// Sets the soil specific heat function to the given spatial scalar function f.
DEFINE_SET_HETEROGENEOUS_SCALAR(SoilSpecificHeat)

/// Sets the saline diffusivity function to a given constant isotropic value.
DEFINE_SET_CONSTANT_ISOTROPIC_TENSOR(SalineDiffusivity)

/// Sets the saline diffusivity function to the given spatial scalar function f that
/// provides an isotropic tensor.
DEFINE_SET_HETEROGENEOUS_ISOTROPIC_TENSOR(SalineDiffusivity)

/// Sets the saline diffusivity function to a given constant diagonal tensor.
DEFINE_SET_CONSTANT_DIAGONAL_TENSOR(SalineDiffusivity)

/// Sets the saline diffusivity function to a given spatial vector function f
/// that computes a diagonal anisotropic tensor.
DEFINE_SET_HETEROGENEOUS_DIAGONAL_TENSOR(SalineDiffusivity)

/// Sets the saline diffusivity function to a given constant tensor.
DEFINE_SET_CONSTANT_TENSOR(SalineDiffusivity)

/// Sets the saline diffusivity function to a given spatial tensor function f that
/// will compute a full anisotropic tensor.
DEFINE_SET_HETEROGENEOUS_TENSOR(SalineDiffusivity)

/// Sets the saline molecular weight function to the given constant value.
DEFINE_SET_CONSTANT_SCALAR(SalineMolecularWeight)

/// Sets the saline molecular weight function to the given spatial scalar function f.
DEFINE_SET_HETEROGENEOUS_SCALAR(SalineMolecularWeight)

#if 0
/*
  Set material properties cell-by-cell
*/

PetscErrorCode TDySetBlockPermeabilityValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[ni], const PetscScalar y[ni]){

  PetscInt i,j;
  PetscInt dim,dim2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  ierr = DMGetDimension(((tdy->discretization)->tdydm)->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  MaterialProp *matprop = tdy->matprop;

  matprop->permeability_is_set = 1;

  for(i=0; i<ni; i++) {
    for(j=0; j<dim2; j++) matprop->K0[ix[i]*dim2 + j] = y[i*dim2+j];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDySetCellPermeability(TDy tdy,PetscInt c,PetscReal *K) {
  PetscInt dim2,i,cStart;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdy,TDY_CLASSID,1);
  MaterialProp *matprop = tdy->matprop;

  matprop->permeability_is_set = 1;

  ierr = DMPlexGetHeightStratum(((tdy->discretization)->tdydm)->dm,0,&cStart,&i); CHKERRQ(ierr);
  ierr = DMGetDimension(((tdy->discretization)->tdydm)->dm,&dim2); CHKERRQ(ierr);
  dim2 *= dim2;
  for(i=0;i<dim2;i++) matprop->K[dim2*(c-cStart)+i] = K[i];
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPorosityValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[ni], const PetscScalar y[ni]){

  PetscInt i;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  if (!ni) PetscFunctionReturn(0);

  MaterialProp *matprop = tdy->matprop;

  matprop->porosity_is_set = 1;

  for(i=0; i<ni; i++) {
    matprop->porosity[ix[i]] = y[i];
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/*
  Get material properties cell-by-cell
*/

PetscErrorCode TDyGetBlockPermeabilityValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd,j;
  PetscInt junkInt, gref;
  PetscInt dim,dim2;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMGetDimension(((tdy->discretization)->tdydm)->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  MaterialProp *matprop = tdy->matprop;

  ierr = DMPlexGetHeightStratum(((tdy->discretization)->tdydm)->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(((tdy->discretization)->tdydm)->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      for(j=0; j<dim2; j++) {
        y[*ni] = matprop->K0[(c-cStart)*dim2 + j];
        *ni += 1;
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyGetPorosityValuesLocal(TDy tdy, PetscInt *ni, PetscScalar y[]){

  PetscInt c,cStart,cEnd;
  PetscInt junkInt, gref;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = DMPlexGetHeightStratum(((tdy->discretization)->tdydm)->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;
  MaterialProp *matprop = tdy->matprop;

  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(((tdy->discretization)->tdydm)->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      y[*ni] = matprop->porosity[c-cStart];
      *ni += 1;
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
#endif
