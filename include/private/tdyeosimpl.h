#if !defined(TDYEOSIMPL_H)
#define TDYEOSIMPL_H

#include <petsc.h>

typedef struct EOS {
  TDyWaterDensityType density_type;
  PetscInt viscosity_type;
  PetscInt enthalpy_type;
} EOS;

PETSC_INTERN PetscErrorCode EOSComputeWaterDensity(EOS*,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode EOSComputeWaterViscosity(EOS*,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode EOSComputeWaterEnthalpy(EOS*,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode EOSComputeSalinityFraction(EOS*,PetscReal,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscReal*);


#endif

