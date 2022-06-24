#include <private/tdymemoryimpl.h>

/// Frees memory allocated by TDyAlloc.
/// @param [out] memory The pointer to the memory to be freed.
PetscErrorCode TDyFree(void *memory) {
  return PetscFree(memory);
}

/// Sets all values in the given memory block to the given integer value.
/// @param size The number of integers in the array
/// @param array The array whose values are to be set
/// @param value The value to which the array's values are set
PetscErrorCode TDySetIntArray(size_t size, PetscInt array[size], PetscInt value) {
  PetscFunctionBegin;
  for (size_t i = 0; i < size; ++i) {
    array[i] = value;
  }
  PetscFunctionReturn(0);
}

/// Sets all values in the given memory block to the given real value.
/// @param size The number of real numbers in the array
/// @param array The array whose values are to be set
/// @param value The value to which the array's values are set
PetscErrorCode TDySetRealArray(size_t size, PetscReal array[size], PetscReal value) {
  PetscFunctionBegin;
  for (size_t i = 0; i < size; ++i) {
    array[i] = value;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_IntegerArray_1D(PetscInt *array_1D, PetscInt ndim_1,
    PetscInt init_value) {

  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    array_1D[i] = init_value;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_IntegerArray_2D(PetscInt **array_2D, PetscInt ndim_1,
                                          PetscInt ndim_2, PetscInt value) {

  PetscInt i,j;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      array_2D[i][j] = value;
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_1D(PetscReal *array_1D, PetscInt ndim_1,
                                       PetscReal value) {

  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    array_1D[i] = value;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_2D(PetscReal **array_2D, PetscInt ndim_1,
                                       PetscInt ndim_2, PetscReal value) {

  PetscInt i,j;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      array_2D[i][j] = value;
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_3D(PetscReal ***array_3D, PetscInt ndim_1,
                                       PetscInt ndim_2, PetscInt ndim_3, PetscReal value) {

  PetscInt i,j,k;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      for (k=0; k<ndim_3; k++) {
        array_3D[i][j][k] = value;
      }
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyInitialize_RealArray_4D(PetscReal ****array_4D, PetscInt ndim_1,
                                       PetscInt ndim_2, PetscInt ndim_3, PetscInt ndim_4, PetscReal value) {

  PetscInt i,j,k,l;
  PetscFunctionBegin;
  for (i=0; i<ndim_1; i++) {
    for (j=0; j<ndim_2; j++) {
      for (k=0; k<ndim_3; k++) {
        for (l=0; l<ndim_4; l++) {
          array_4D[i][j][k][l] = value;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_IntegerArray_1D(PetscInt **array_1D, PetscInt ndim_1) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TDyAlloc(ndim_1*sizeof(PetscInt), array_1D); CHKERRQ(ierr);
  ierr = TDyInitialize_IntegerArray_1D(*array_1D, ndim_1, -1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_IntegerArray_2D(PetscInt ***array_2D, PetscInt ndim_1,
                                     PetscInt ndim_2) {

  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TDyAlloc(ndim_1*sizeof(PetscInt *), array_2D); CHKERRQ(ierr);
  for(i=0; i<ndim_1; i++) {
    ierr = TDyAlloc(ndim_2*sizeof(PetscInt ), &((*array_2D)[i])); CHKERRQ(ierr);
  }
  ierr = TDyInitialize_IntegerArray_2D(*array_2D, ndim_1, ndim_2, -1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_1D(PetscReal **array_1D, PetscInt ndim_1) {
  return TDyAlloc(ndim_1*sizeof(PetscReal), array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_2D(PetscReal ***array_2D, PetscInt ndim_1,
                                     PetscInt ndim_2) {

  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TDyAlloc(ndim_1*sizeof(PetscReal *), array_2D); CHKERRQ(ierr);
  for(i=0; i<ndim_1; i++) {
    ierr = TDyAlloc(ndim_2*sizeof(PetscReal ), &((*array_2D)[i])); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_3D(PetscReal ****array_3D, PetscInt ndim_1,
                                     PetscInt ndim_2, PetscInt ndim_3) {

  PetscInt i,j;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TDyAlloc(ndim_1*sizeof(PetscReal **), array_3D); CHKERRQ(ierr);
  for(i=0; i<ndim_1; i++) {
    ierr = TDyAlloc(ndim_2*sizeof(PetscReal *), &((*array_3D)[i])); CHKERRQ(ierr);
    for(j=0; j<ndim_2; j++) {
      ierr = TDyAlloc(ndim_3*sizeof(PetscReal), &((*array_3D)[i][j])); CHKERRQ(ierr);
    }
  }
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_RealArray_4D(PetscReal *****array_4D, PetscInt ndim_1,
                                     PetscInt ndim_2, PetscInt ndim_3, PetscInt ndim_4) {

  PetscInt i,j,k;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TDyAlloc(ndim_1*sizeof(PetscReal ***), array_4D); CHKERRQ(ierr);
  for(i=0; i<ndim_1; i++) {
    ierr = TDyAlloc(ndim_2*sizeof(PetscReal **), &((*array_4D)[i])); CHKERRQ(ierr);
    for(j=0; j<ndim_2; j++) {
      ierr = TDyAlloc(ndim_3*sizeof(PetscReal *), &((*array_4D)[i][j])); CHKERRQ(ierr);
      for(k=0; k<ndim_3; k++) {
        ierr = TDyAlloc(ndim_4*sizeof(PetscReal), &((*array_4D)[i][j][k])); CHKERRQ(ierr);
      }
    }
  }
  ierr = TDyInitialize_RealArray_4D(*array_4D, ndim_1, ndim_2, ndim_3, ndim_4, 0.0);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_RealArray_1D(PetscReal *array_1D) {
  return TDyFree(array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_IntegerArray_1D(PetscInt *array_1D) {
  return TDyFree(array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_IntegerArray_2D(PetscInt **array_2D, PetscInt ndim_1) {

  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    ierr = TDyFree(array_2D[i]); CHKERRQ(ierr);
  }
  ierr = TDyFree(array_2D); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_RealArray_2D(PetscReal **array_2D, PetscInt ndim_1) {

  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    ierr = TDyFree(array_2D[i]); CHKERRQ(ierr);
  }
  ierr = TDyFree(array_2D); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_RealArray_3D(PetscReal ***array_3D, PetscInt ndim_1, PetscInt ndim_2) {

  PetscErrorCode ierr;
  PetscInt i,j;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for(j=0; j<ndim_2; j++) {
      ierr = TDyFree(array_3D[i][j]); CHKERRQ(ierr);
    }
    ierr = TDyFree(array_3D[i]); CHKERRQ(ierr);
  }
  ierr = TDyFree(array_3D); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDeallocate_RealArray_4D(PetscReal ****array_4D, PetscInt ndim_1, PetscInt ndim_2, PetscInt ndim_3) {

  PetscErrorCode ierr;
  PetscInt i,j,k;
  PetscFunctionBegin;
  for(i=0; i<ndim_1; i++) {
    for(j=0; j<ndim_2; j++) {
      for(k=0; k<ndim_3; k++) {
        ierr = TDyFree(array_4D[i][j][k]); CHKERRQ(ierr);
      }
      ierr = TDyFree(array_4D[i][j]); CHKERRQ(ierr);
    }
    ierr = TDyFree(array_4D[i]); CHKERRQ(ierr);
  }
  ierr = TDyFree(array_4D); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyCell_1D(PetscInt ndim_1, TDyCell **array_1D) {
  return TDyAlloc(ndim_1*sizeof(TDyCell), array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyVertex_1D(PetscInt ndim_1, TDyVertex **array_1D) {
  return TDyAlloc(ndim_1*sizeof(TDyVertex), array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyEdge_1D(PetscInt ndim_1, TDyEdge **array_1D) {
  return TDyAlloc(ndim_1*sizeof(TDyEdge), array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyFace_1D(PetscInt ndim_1, TDyFace **array_1D) {
  return TDyAlloc(ndim_1*sizeof(TDyFace), array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDySubcell_1D(PetscInt ndim_1, TDySubcell **array_1D) {
  return TDyAlloc(ndim_1*sizeof(TDySubcell), array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyVector_1D(PetscInt ndim_1, TDyVector **array_1D) {
  return TDyAlloc(ndim_1*sizeof(TDyVector), array_1D);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyAllocate_TDyCoordinate_1D(PetscInt ndim_1, TDyCoordinate **array_1D) {
  return TDyAlloc(ndim_1*sizeof(TDyCoordinate), array_1D);
}
