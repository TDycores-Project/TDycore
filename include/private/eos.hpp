#ifndef TDYCORE_EOS_HPP
#define TDYCORE_EOS_HPP

#include <petsc.h>

struct EOS {
  TDyWaterDensityType density_type;
  PetscInt viscosity_type;
  PetscInt enthalpy_type;

  void ComputeWaterDensity(PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*) const;
  void ComputeWaterViscosity(PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*) const;
  void ComputeWaterEnthalpy(PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*) const;
  void ComputeSalinityFraction(PetscReal,PetscReal,PetscReal,PetscReal*) const;
};

#endif

