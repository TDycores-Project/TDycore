#if !defined(TDYMEMORYIMPL_H)
#define TDYMEMORYIMPL_H

#include <petsc.h>
#include <private/tdymeshimpl.h>

/// @def TDyAlloc
/// Allocates a chunk of zeroed memory of the given size (in bytes).
/// @param [in] size The size of the requested allocation.
/// @param [out] result A pointer to the requested memory chunk.
#define TDyAlloc(size,result) PetscCalloc1(size,result)

/// @def TDyRealloc
/// Resizes the given chunk of memory to the new requested size.
/// @param [in] size The size of the requested allocation.
/// @param [inout] memory A pointer to a previously allocated memory chunk.
#define TDyRealloc(size,memory) PetscRealloc(size,memory)
PETSC_INTERN PetscErrorCode TDyFree(void*);
PETSC_INTERN PetscErrorCode TDySetIntArray(size_t,PetscInt[],PetscInt);
PETSC_INTERN PetscErrorCode TDySetRealArray(size_t,PetscReal[],PetscReal);

/// @def TDY_DECLARE_2D_ARRAY
/// Declares a 2D array variable that uses specified storage
/// @param type The (primitive) data type of the array
/// @param array_var The declared multidimensional array
/// @param storage A "flat" array with enough room to fit all the data in array_var
/// @param dim1 The dimension of the array's first component
/// @param dim2 The dimension of the array's first component
#define TDY_DECLARE_2D_ARRAY(type, array_var, storage, dim1, dim2) \
type (*array_var)[dim2] = (void*)storage

/// @def TDY_DECLARE_3D_ARRAY
/// Declares a 3D array variable that uses specified storage
/// @param type The (primitive) data type of the array
/// @param array_var The declared multidimensional array
/// @param storage A "flat" array with enough room to fit all the data in array_var.
/// @param dim1 The dimension of the array's first component
/// @param dim2 The dimension of the array's second component
/// @param dim3 The dimension of the array's third component
#define TDY_DECLARE_3D_ARRAY(type, array_var, storage, dim1, dim2, dim3) \
type (*array_var)[dim2][dim3] = (void*)storage

/// @def TDY_DECLARE_4D_ARRAY
/// Declares a 4D array variable that uses specified storage
/// @param type The (primitive) data type of the array
/// @param array_var The declared multidimensional array
/// @param storage A "flat" array with enough room to fit all the data in array_var
/// @param dim1 The dimension of the array's first component
/// @param dim2 The dimension of the array's second component
/// @param dim3 The dimension of the array's third component
/// @param dim4 The dimension of the array's fourth component
#define TDY_DECLARE_4D_ARRAY(type, array_var, storage, dim1, dim2, dim3, dim4) \
type (*array_var)[dim2][dim3][dim4] = (void*)storage

/// @def TDY_DECLARE_5D_ARRAY
/// Declares a 5D array variable that uses specified storage
/// @param type The (primitive) data type of the array
/// @param array_var The declared multidimensional array
/// @param storage A "flat" array with enough room to fit all the data in array_var
/// @param dim1 The dimension of the array's first component
/// @param dim2 The dimension of the array's second component
/// @param dim3 The dimension of the array's third component
/// @param dim4 The dimension of the array's fourth component
/// @param dim5 The dimension of the array's fifth component
#define TDY_DECLARE_5D_ARRAY(type, array_var, storage, dim1, dim2, dim3, dim4, dim5) \
type (*array_var)[dim2][dim3][dim4][dim5] = (void*)storage

/// @def TDY_DECLARE_6D_ARRAY
/// Declares a 6D array variable that uses specified storage
/// @param type The (primitive) data type of the array
/// @param array_var The declared multidimensional array
/// @param storage A "flat" array with enough room to fit all the data in array_var
/// @param dim1 The dimension of the array's first component
/// @param dim2 The dimension of the array's second component
/// @param dim3 The dimension of the array's third component
/// @param dim4 The dimension of the array's fourth component
/// @param dim5 The dimension of the array's fifth component
/// @param dim6 The dimension of the array's sixth component
#define TDY_DECLARE_6D_ARRAY(type, array_var, storage, dim1, dim2, dim3, dim4, dim5, dim6) \
type (*array_var)[dim2][dim3][dim4][dim5][dim6] = (void*)storage

/// @def TDY_DECLARE_7D_ARRAY
/// Declares a 7D array variable that uses specified storage
/// @param type The (primitive) data type of the array
/// @param array_var The declared multidimensional array
/// @param storage A "flat" array with enough room to fit all the data in array_var
/// @param dim1 The dimension of the array's first component
/// @param dim2 The dimension of the array's second component
/// @param dim3 The dimension of the array's third component
/// @param dim4 The dimension of the array's fourth component
/// @param dim5 The dimension of the array's fifth component
/// @param dim6 The dimension of the array's sixth component
/// @param dim7 The dimension of the array's seventh component
#define TDY_DECLARE_7D_ARRAY(type, array_var, storage, dim1, dim2, dim3, dim4, dim5, dim6, dim7) \
type (*array_var)[dim2][dim3][dim4][dim5][dim6][dim7] = (void*)storage

// These functions will be removed in the future.
PETSC_INTERN PetscErrorCode TDyInitialize_IntegerArray_1D(PetscInt*,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyInitialize_IntegerArray_2D(PetscInt**,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_1D(PetscReal*,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_2D(PetscReal**,PetscInt,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_3D(PetscReal***,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyInitialize_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TDyAllocate_IntegerArray_1D(PetscInt**,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_IntegerArray_2D(PetscInt***,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_1D(PetscReal**,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_2D(PetscReal***,PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_3D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_RealArray_4D(PetscReal*****,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_1D(PetscReal*);
PETSC_INTERN PetscErrorCode TDyDeallocate_IntegerArray_1D(PetscInt*);
PETSC_INTERN PetscErrorCode TDyDeallocate_IntegerArray_2D(PetscInt**,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_2D(PetscReal**,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_3D(PetscReal***,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyDeallocate_RealArray_4D(PetscReal****,PetscInt,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyCell_1D(PetscInt,TDyCell**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyVertex_1D(PetscInt,TDyVertex**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyEdge_1D(PetscInt,TDyEdge**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyFace_1D(PetscInt,TDyFace**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDySubcell_1D(PetscInt,TDySubcell**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyVector_1D(PetscInt,TDyVector**);
PETSC_INTERN PetscErrorCode TDyAllocate_TDyCoordinate_1D(PetscInt,TDyCoordinate**);

#endif

