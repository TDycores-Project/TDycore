#if !defined(TDYOPTIONS_H)
#define TDYOPTIONS_H

#include <petsc.h>

typedef struct {

  // Model settings
  PetscReal gravity_constant;
  TDyMode mode;

  // Numerics settings
  TDyDiscretization discretization;

  // EOS settings
  TDyWaterDensityType rho_type;
  PetscInt mu_type;
  PetscInt enthalpy_type;

  // Constant material properties
  PetscReal porosity;
  PetscReal permeability;
  PetscReal thermal_conductivity;
  PetscReal residual_saturation;
  PetscReal soil_density;
  PetscReal soil_specific_heat;

  // Characteristic curve parameters
  PetscReal gardner_n;
  PetscReal vangenuchten_m;
  PetscReal vangenuchten_alpha;
  PetscReal mualem_poly_x0;
  PetscReal mualem_poly_x1;
  PetscReal mualem_poly_x2;
  PetscReal mualem_poly_dx;

  // Constant boundary values
  PetscReal boundary_pressure;
  PetscReal boundary_temperature;
  PetscReal boundary_velocity; // (normal component)

  // Initial conditions
  PetscBool init_with_random_field;
  PetscBool init_from_file;
  char init_file[PETSC_MAX_PATH_LEN];

  // Mesh-related options
  PetscBool read_mesh;
  PetscBool read_pflotran_mesh;
  char mesh_file[PETSC_MAX_PATH_LEN];

  // I/O settings
  PetscBool output_mesh;
  PetscBool regression_testing;

  // Timers enabled?
  PetscBool enable_timers;
} TDyOptions;

#endif
