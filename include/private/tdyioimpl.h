#if !defined(TDYIOIMPL_H)
#define TDYIOIMPL_H

#include <petsc.h>

struct _p_TDyIO {
  PetscBool io_process;
  PetscBool print_intermediate;
  char filename[PETSC_MAX_PATH_LEN];
  char zonalVarNames[2][PETSC_MAX_PATH_LEN];
  int num_vars;
  TDyIOFormat format;
  int num_times;
  char permeability_filename[PETSC_MAX_PATH_LEN];
  char permeability_dataset[PETSC_MAX_PATH_LEN];
  char porosity_filename[PETSC_MAX_PATH_LEN];
  char porosity_dataset[PETSC_MAX_PATH_LEN];
  char ic_filename[PETSC_MAX_PATH_LEN];
  char ic_dataset[PETSC_MAX_PATH_LEN];
};

PETSC_INTERN PetscErrorCode TDyIOCreate(TDyIO*);
PETSC_INTERN PetscErrorCode TDyIOReadPermeability(TDy);
PETSC_INTERN PetscErrorCode TDyIOReadPorosity(TDy);
PETSC_INTERN PetscErrorCode TDyIOReadIC(TDy);
PETSC_INTERN PetscErrorCode TDyIOInitializeExodus(char*,char*[],DM ,int);
PETSC_INTERN PetscErrorCode TDyIOAddExodusTime(char*,PetscReal,DM,TDyIO);
PETSC_INTERN PetscErrorCode TDyIOWriteExodusVar(char*,Vec,char*,TDyIO,PetscReal);
PETSC_INTERN PetscErrorCode TDyIOInitializeHDF5(char*,DM);
PETSC_INTERN PetscErrorCode TDyIOWriteHDF5Var(char*,DM,Vec,char*,PetscReal);
PETSC_INTERN PetscErrorCode TDyIOWriteAsciiViewer(Vec,PetscReal,char*);
PETSC_INTERN PetscErrorCode TDyIOWriteXMFHeader(PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyIOWriteXMFAttribute(char*,char*,PetscInt);
PETSC_INTERN PetscErrorCode TDyIOWriteXMFFooter();
PETSC_INTERN PetscErrorCode TDyIODestroy(TDyIO*);

#endif

