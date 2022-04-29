#if !defined(TDYDMIMPL_H)
#define TDYDMIMPL_H

#include <petsc.h>


typedef struct {
  IS is_GhostedCells_in_LocalOrder;
  IS is_GhostedCells_in_PetscOrder;

  IS is_LocalCells_in_LocalOrder;
  IS is_LocalCells_in_PetscOrder;

  IS is_GhostCells_in_LocalOrder;
  IS is_GhostCells_in_PetscOrder;

  IS is_LocalCells_to_NaturalCells;

  VecScatter scatter_LocalCells_to_NaturalCells;
  VecScatter scatter_GlobalCells_to_LocalCells;
  VecScatter scatter_LocalCells_to_LocalCells;
  VecScatter scatter_GlobalCells_to_NaturalCells;

  ISLocalToGlobalMapping mapping_LocalCells_to_NaturalCells;

} TDyUGDM;

typedef struct {
  DM dm;
  TDyUGDM ugdm;

} TDyDM;

#endif