#include <private/tdycoreimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
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

  ierr = PetscCalloc(sizeof(MaterialProp), matprop); CHKERRQ(ierr);
  (*matprop)->dim = dim;

  PetscFunctionReturn(0);
}

/// Destroys the given instance (and any context pointers managed by it).
/// @param [inout] matprop A MaterialProp instance
PetscErrorCode MaterialPropDestroy(MaterialProp *matprop) {
  PetscFunctionBegin;
  MaterialPropSetPorosity(matprop, NULL, NULL, NULL);
  MaterialPropSetPermeability(matprop, NULL, NULL, NULL);
  MaterialPropSetThermalConductivity(matprop, NULL, NULL, NULL);
  MaterialPropSetResidualSaturation(matprop, NULL, NULL, NULL);
  MaterialPropSetSoilDensity(matprop, NULL, NULL, NULL);
  MaterialPropSetSoilSpecificHeat(matprop, NULL, NULL, NULL);

  PetscFunctionReturn(0);
}

/// Sets the function used to compute the permeability.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that computes the permeability at a given number of points
/// @param [in] dtor A function that destroys the context when matprop is destroyed (can be NULL).
PetscErrorCode MaterialPropSetPermeability(MaterialProp *matprop, void *context,
                                           PetscErrorCode(*f)(void*,PetscInt,PetscReal*,PetscReal*),
                                           void (*dtor)(void*)) {
  PetscErrorCode ierr;
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
                                       void (*dtor)(void*)) {
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
                                                  void (*dtor)(void*)) {
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
                                                 void (*dtor)(void*)) {
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
                                          void (*dtor)(void*)) {
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
                                               void (*dtor)(void*)) {
  PetscFunctionBegin;
  if (matprop->soil_specific_heat_context && matprop->soil_specific_heat_dtor)
    matprop->soil_specific_heat_dtor(matprop->soil_specific_heat_context);
  matprop->soil_specific_heat_context = context;
  matprop->compute_soil_specific_heat = f;
  matprop->soil_specific_heat_dtor = dtor;
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
    ierr = matprop->compute_thermal_conductivity(matprop->soil_density_context,
                                                 n, x, conductivity); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/// Computes the residual saturation at the given set of points (if the material
/// properties have a corresponding function).
/// @param [in] matprop A MaterialProp instance
/// @param [in] n The number of points
/// @param [in] x The coordinates of the n points
/// @param [out] saturation The values of the saturation at each point
PetscErrorCode MaterialPropComputeResidualSaturation(MaterialProp *matprop,
                                                     PetscInt n, PetscReal *x,
                                                     PetscReal *saturation) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (matprop->compute_residual_saturation) {
    ierr = matprop->compute_residual_saturation(matprop->soil_density_context,
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
static PetscErrorCode CreateConstantContext(int dim, PetscReal value[9],
                                            void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper;
  ierr = PetscMalloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  wrapper->dim = dim;
  memcpy(wrapper->value, value, 9*sizeof(PetscReal));
  wrapper->func = NULL;
  *context = wrapper;
  PetscFunctionReturn(0);
}

// Generic constructor for contexts for spatial functions in this file.
static PetscErrorCode CreateSpatialContext(int dim,
                                           void (*func)(PetscInt, PetscReal*, PetscReal*),
                                           void **context) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  WrapperStruct *wrapper;
  ierr = PetscMalloc(sizeof(WrapperStruct), &wrapper); CHKERRQ(ierr);
  wrapper->dim = func;
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
  for (int i = 0; i < n; ++i) {
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
  for (int i = 0; i < n; ++i) {
    for(int j = 0; j < dim; ++j) {
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
  for (int i = 0; i < n; ++i) {
    for(int j = 0; j < dim; ++j) {
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
  for (int i = 0; i < n; ++i) {
    for(int j = 0; j < dim; ++j) {
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
  for (int i = 0; i < n; ++i) {
    for(int j = 0; j < dim; ++j) {
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
  for (int i = 0; i < n; ++i) {
    for(int j = 0; j < dim2; ++j) {
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

// Generic destructor for contexts for functions in this file. Called when
// the corresponding MaterialProp instance is destroyed.
static void DestroyContext(void* context) {
  PetscFree(context);
}

/// Sets the porosity function to the given constant value.
PetscErrorCode MaterialPropSetConstantPorosity(MaterialProp *matprop,
                                               PetscReal porosity) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = porosity;
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPorosity(matprop, context, ConstantScalarWrapperFunction,
                                 DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the porosity function to the given spatial scalar function f.
PetscErrorCode MaterialPropSetHeterogeneousPorosity(MaterialProp *matprop,
                                                    SpatialScalarFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPorosity(matprop, context, ScalarWrapperFunction,
                                 DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability function to a given constant value that provides an
/// isotropic permeability.
PetscErrorCode MaterialPropSetConstantIsotropicPermeability(MaterialProp *matprop,
                                                            PetscReal perm) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = perm;
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPermeability(matprop, context, ConstantIsotropicTensorWrapperFunction,
                                     DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability function to the given spatial scalar function f that
/// provides an isotropic permeability.
PetscErrorCode MaterialPropSetHeterogeneousIsotropicPermeability(MaterialProp *matprop,
                                                                 SpatialScalarFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPermeability(matprop, context, IsotropicTensorWrapperFunction,
                                     DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability function to a given constant diagonal permeability.
PetscErrorCode MaterialPropSetConstantDiagonalPermeability(MaterialProp *matprop,
                                                           PetscReal perm[3]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = perm[0]; value[1] = perm[1]; value[2] = perm[2];
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPermeability(matprop, context, ConstantDiagonalTensorWrapperFunction,
                                     DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability function to a given spatial vector function f that
/// will compute a diagonal anisotropic permeability tensor.
PetscErrorCode MaterialPropSetHeterogeneousDiagonalPermeability(MaterialProp *matprop,
                                                                SpatialVectorFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPermeability(matprop, context, DiagonalTensorWrapperFunction,
                                     DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability function to a given constant tensor permeability.
PetscErrorCode MaterialPropSetConstantTensorPermeability(MaterialProp *matprop,
                                                         PetscReal perm[9]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateConstantContext(matprop->dim, perm, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPermeability(matprop, context, ConstantFullTensorWrapperFunction,
                                     DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the permeability function to a given spatial tensor function f that
/// will compute a full anisotropic permeability tensor.
PetscErrorCode MaterialPropSetHeterogeneousTensorPermeability(MaterialProp *matprop,
                                                              SpatialTensorFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetPermeability(matprop, context, FullTensorWrapperFunction,
                                     DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity function to a given constant value that
/// provides an isotropic conductivity.
PetscErrorCode MaterialPropSetConstantIsotropicThermalConductivity(MaterialProp *matprop,
                                                                   PetscReal conductivity) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = conductivity;
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetThermalConductivity(matprop, context, ConstantIsotropicTensorWrapperFunction,
                                            DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity function to the given spatial scalar function f that
/// provides an isotropic conductivity.
PetscErrorCode MaterialPropSetHeterogeneousIsotropicThermalConductivity(MaterialProp *matprop,
                                                                        SpatialScalarFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetThermalConductivity(matprop, context, IsotropicTensorWrapperFunction,
                                            DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity function to a given constant diagonal conductivity.
PetscErrorCode MaterialPropSetConstantDiagonalThermalConductivity(MaterialProp *matprop,
                                                                  PetscReal conductivity[3]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = conductivity[0]; value[1] = conductivity[1]; value[2] = conductivity[2];
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetThermalConductivity(matprop, context, ConstantDiagonalTensorWrapperFunction,
                                             DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity function to a given spatial vector function f that
/// will compute a diagonal anisotropic conductivity tensor.
PetscErrorCode MaterialPropSetHeterogeneousDiagonalThermalConductivity(MaterialProp *matprop,
                                                                       SpatialVectorFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetThermalConductivity(matprop, context, DiagonalTensorWrapperFunction,
                                            DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity function to a given constant tensor permeability.
PetscErrorCode MaterialPropSetConstantTensorThermalConductivity(MaterialProp *matprop,
                                                                PetscReal conductivity[9]) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateConstantContext(matprop->dim, conductivity, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetThermalConductivity(matprop, context, ConstantFullTensorWrapperFunction,
                                            DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the thermal conductivity function to a given spatial tensor function f that
/// will compute a full anisotropic conductivity tensor.
PetscErrorCode MaterialPropSetHeterogeneousTensorThermalConductivity(MaterialProp *matprop,
                                                                     SpatialTensorFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetThermalConductivity(matprop, context, FullTensorWrapperFunction,
                                            DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the residual saturation function to the given constant value.
PetscErrorCode MaterialPropSetConstantResidualSaturation(MaterialProp *matprop,
                                                         PetscReal saturation) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = saturation;
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetResidualSaturation(matprop, context, ConstantScalarWrapperFunction,
                                           DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the residual saturation function to the given spatial scalar function f.
PetscErrorCode MaterialPropSetHeterogeneousResidualSaturation(MaterialProp *matprop,
                                                              SpatialScalarFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetResidualSaturation(matprop, context, ScalarWrapperFunction,
                                           DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the soil density function to the given constant value.
PetscErrorCode MaterialPropSetConstantSoilDensity(MaterialProp *matprop,
                                                  PetscReal density) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = density;
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetSoilDensity(matprop, context, ConstantScalarWrapperFunction,
                                    DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the soil density function to the given spatial scalar function f.
PetscErrorCode MaterialPropSetHeterogeneousSoilDensity(MaterialProp *matprop,
                                                       SpatialScalarFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetSoilDensity(matprop, context, ScalarWrapperFunction,
                                    DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the soil specific heat function to the given constant value.
PetscErrorCode MaterialPropSetConstantSoilSpecificHeat(MaterialProp *matprop,
                                                       PetscReal specific_heat) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  PetscReal value[9];
  value[0] = specific_heat;
  ierr = CreateConstantContext(matprop->dim, value, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetSoilSpecificHeat(matprop, context, ConstantScalarWrapperFunction,
                                                                                                         DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the soil specific heat function to the given spatial scalar function f.
PetscErrorCode MaterialPropSetHeterogeneousSoilSpecificHeat(MaterialProp *matprop,
                                                            SpatialScalarFunction f) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  void *context;
  ierr = CreateSpatialContext(matprop->dim, f, &context); CHKERRQ(ierr);
  ierr = MaterialPropSetSoilSpecificHeat(matprop, context, ScalarWrapperFunction,
                                         DestroyContext); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
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

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&i); CHKERRQ(ierr);
  ierr = DMGetDimension(tdy->dm,&dim2); CHKERRQ(ierr);
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

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  MaterialProp *matprop = tdy->matprop;

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;

  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
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

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  *ni = 0;
  MaterialProp *matprop = tdy->matprop;

  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(tdy->dm,c,&gref,&junkInt); CHKERRQ(ierr);
    if (gref>=0) {
      y[*ni] = matprop->porosity[c-cStart];
      *ni += 1;
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
#endif
