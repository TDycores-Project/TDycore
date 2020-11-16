#if !defined(TDYREGRESSIONIMPL_H)
#define TDYREGRESSIONIMPL_H

#include <petsc.h>
#include "tdycore.h"

typedef struct {
  char filename[PETSC_MAX_PATH_LEN];
  PetscInt num_cells_per_process;
  PetscInt *cells_per_process_natural_ids;
  Vec cells_per_process_vec;
  VecScatter scatter_cells_per_process_gtos;
} TDyRegression;

PETSC_INTERN PetscErrorCode TDyRegressionInitialize(TDy);
PETSC_INTERN PetscErrorCode TDyRegressionOutput(TDy,Vec);

#endif
