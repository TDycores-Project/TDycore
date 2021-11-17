#if !defined(TDYOPTIONS_H)
#define TDYOPTIONS_H

#include <petsc.h>

typedef struct {

  // Model settings
  PetscReal gravity_constant;
  TDyMode mode;
  TDyWaterDensityType rho_type;
  PetscInt mu_type;
  PetscInt enthalpy_type;

  // Numerics settings
  TDyMethod method;
  TDyQuadratureType qtype;
  PetscInt mpfao_gmatrix_method;
  PetscInt mpfao_bc_type;
  PetscBool tpf_allow_all_meshes;

  // Constant material properties
  PetscReal porosity;
  PetscReal permeability;
  PetscReal soil_density;
  PetscReal soil_specific_heat;
  PetscReal thermal_conductivity;
  PetscReal molecular_weight;
   PetscReal diffusion;

  // Characteristic curve parameters
  PetscReal residual_saturation;
  PetscReal gardner_n;
  PetscReal vangenuchten_m;
  PetscReal vangenuchten_alpha;

  // Constant boundary values
  PetscReal boundary_pressure;
  PetscReal boundary_temperature;
  PetscReal boundary_velocity; // (normal component)

  // Initial conditions
  PetscBool init_with_random_field;
  PetscBool init_from_file;
  char init_file[PETSC_MAX_PATH_LEN];

  // Mesh-related options
  PetscBool generate_mesh;
  PetscBool read_mesh;
  char mesh_file[PETSC_MAX_PATH_LEN];

  PetscBool output_geom_attributes;
  PetscBool read_geom_attributes;
  char geom_attributes_file[PETSC_MAX_PATH_LEN];

  // I/O settings
  PetscBool output_mesh;
  PetscBool regression_testing;

  // Timers enabled?
  PetscBool enable_timers;
} TDyOptions;

#endif
