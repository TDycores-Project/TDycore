#include <private/tdycoreimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <tdytimers.h>
#include <private/tdyoptions.h>

// FIXME: This type needs to be reimagined as a description of material
// FIXME: properties, with data owned by the implementations.

/// Creates a new MaterialProp instance, allocating storage for data.
/// @param [in] dim The dimension of the space occupied by the materials
/// @param [in] npoints The number of points at which material properties
///                     are stored.
/// @param [in] points An array of points, not managed by the instance.
/// @param [out] matprop Storage for the new instance
PetscErrorCode MaterialPropCreate(PetscInt dim, PetscInt npoints,
                                  PetscReal *points, MaterialProp **matprop) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  *matprop = (MaterialProp *)malloc(sizeof(MaterialProp));
  (*matprop)->dim = dim;
  (*matprop)->num_points = npoints;
  (*matprop)->points = points;

  ierr = PetscMalloc(npoints*ndim*ndim*sizeof(PetscReal),&(*matprop)->K); CHKERRQ(ierr);
  ierr = PetscMalloc(npoints*ndim*ndim*sizeof(PetscReal),&(*matprop)->K0); CHKERRQ(ierr);
  ierr = PetscMalloc(npoints*ndim*ndim*sizeof(PetscReal),&(*matprop)->Kappa); CHKERRQ(ierr);
  ierr = PetscMalloc(npoints*ndim*ndim*sizeof(PetscReal),&(*matprop)->Kappa0); CHKERRQ(ierr);

  ierr = PetscMalloc(npoints*sizeof(PetscReal),&(*matprop)->porosity); CHKERRQ(ierr);
  ierr = PetscMalloc(npoints*sizeof(PetscReal),&(*matprop)->Cr); CHKERRQ(ierr);
  ierr = PetscMalloc(npoints*sizeof(PetscReal),&(*matprop)->rhosoil); CHKERRQ(ierr);

  (*matprop)->porosity_context = NULL;
  (*matprop)->permeability_context = NULL;
  (*matprop)->thermal_conductivity_context = NULL;
  (*matprop)->residual_saturation_context = NULL;
  (*matprop)->soil_density_context = NULL;
  (*matprop)->soil_specific_heat = NULL;

  (*matprop)->compute_porosity = NULL;
  (*matprop)->compute_permeability = NULL;
  (*matprop)->compute_thermal_conductivity = NULL;
  (*matprop)->compute_residual_saturation = NULL;
  (*matprop)->compute_soil_density = NULL;
  (*matprop)->compute_soil_specific_heat = NULL;

  (*matprop)->updated = PETSC_FALSE;

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPropDestroy(MaterialProp *matprop) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  if (matprop->K       ) {ierr = PetscFree(matprop->K       ); CHKERRQ(ierr);}
  if (matprop->K0      ) {ierr = PetscFree(matprop->K0      ); CHKERRQ(ierr);}
  if (matprop->Kappa   ) {ierr = PetscFree(matprop->Kappa   ); CHKERRQ(ierr);}
  if (matprop->Kappa0  ) {ierr = PetscFree(matprop->Kappa0  ); CHKERRQ(ierr);}
  if (matprop->porosity) {ierr = PetscFree(matprop->porosity); CHKERRQ(ierr);}
  if (matprop->Cr      ) {ierr = PetscFree(matprop->Cr      ); CHKERRQ(ierr);}
  if (matprop->rhosoil ) {ierr = PetscFree(matprop->rhosoil ); CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

/// Sets the permeability values at all points within the given material
/// properties by calling the given function f with the given context.
/// @param [in] matprop A MaterialProp instance
/// @param [in] context A context pointer to be passed to f
/// @param [in] f A function that updates of the specified number of values
PetscErrorCode MaterialPropSetPermeability(MaterialProp *matprop, void *context,
                                           PetscErrorCode(*f)(void*,int,PetscReal*,PetscReal*)) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = f(context, matprop->num_points, 
  matprop->compute_permeability = f;
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPropSetPorosity(MaterialProp *matprop, void *context,
                                       PetscErrorCode(*f)(void*,int,PetscReal*,PetscReal*)) {
  PetscFunctionBegin;
  matprop->porosity_context = context;
  matprop->compute_porosity = f;
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPropSetThermalConductivity(MaterialProp *matprop, void *context,
                                                  PetscErrorCode(*f)(void*,int,PetscReal*,PetscReal*)) {
  PetscFunctionBegin;
  matprop->thermal_conductivity_context = context;
  matprop->compute_thermal_conductivity = f;
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPropSetResidualSaturation(MaterialProp *matprop, void *context,
                                                 PetscErrorCode(*f)(void*,int,PetscReal*,PetscReal*)) {
  PetscFunctionBegin;
  matprop->residual_saturation_context = context;
  matprop->compute_residual_saturation = f;
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPropSetSoilDensity(MaterialProp *matprop, void *context,
                                          PetscErrorCode(*f)(void*,int,PetscReal*,PetscReal*)) {
  PetscFunctionBegin;
  tdy->matprop->soil_density_is_set = 1;
  if (f) tdy->ops->computesoildensity = f;
  if (ctx) tdy->soildensityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPropSetSoilSpecificHeat(MaterialProp *matprop, void *context,
                                               PetscErrorCode(*f)(void*,int,PetscReal*,PetscReal*)) {
  PetscFunctionBegin;
  tdy->matprop->soil_specific_heat_is_set = 1;
  if (f) tdy->ops->computesoilspecificheat = f;
  if (ctx) tdy->soilspecificheatctx = ctx;
  PetscFunctionReturn(0);
}

/// Returns true if the porosity has been set, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasPorosity(MaterialProp* matprop) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->has_porosity);
}

/// Returns true if the permeability has been set, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasPermeability(MaterialProp*) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->has_permeability);
}

/// Returns true if the thermal conductivity has been set, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasThermalConductivity(MaterialProp*) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->has_thermal_conductivity);
}

/// Returns true if the residual saturation has been set, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasResidualSaturation(MaterialProp*) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->has_residual_saturation);
}

/// Returns true if the soil density has been set, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasSoilDensity(MaterialProp*) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->has_soil_density);
}

/// Returns true if the soil specific heat has been set, false otherwise.
/// @param [in] matprop A MaterialProp instance
PetscBool MaterialPropHasSoilSpecificHeat(MaterialProp*) {
  PetscFunctionBegin;
  PetscFunctionReturn(matprop->has_soil_specific_heat);
}

/*
  Material properties set by SpatialFunction
*/

PetscErrorCode TDySetPermeabilityScalar(TDy tdy,SpatialFunction f) {
  PetscInt dim,dim2,i,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  MaterialProp *matprop = tdy->matprop;

  matprop->permeability_is_set = 1;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(matprop->K,sizeof(PetscReal)*dim2*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    f(&(tdy->X[dim*c]),&(matprop->K[dim2*(c-cStart)]));
    for(i=1; i<dim; i++) {
      matprop->K[dim2*(c-cStart)+i*dim+i] = matprop->K[dim2*(c-cStart)];
    }
  }
  ierr = PetscMemcpy(matprop->K0,matprop->K,sizeof(PetscReal)*dim2*(cEnd-cStart));
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPermeabilityDiagonal(TDy tdy,SpatialFunction f) {
  PetscInt dim,dim2,i,c,cStart,cEnd;
  PetscReal val[3];
  MaterialProp *matprop = tdy->matprop;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  matprop->permeability_is_set = 1;

  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(matprop->K,sizeof(PetscReal)*dim2*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    f(&(tdy->X[dim*c]),&(val[0]));
    for(i=0; i<dim; i++) {
      matprop->K[dim2*(c-cStart)+i*dim+i] = val[i];
    }
  }
  ierr = PetscMemcpy(matprop->K0,matprop->K,sizeof(PetscReal)*dim2*(cEnd-cStart));
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPermeabilityTensor(TDy tdy,SpatialFunction f) {
  PetscInt dim,dim2,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  MaterialProp *matprop = tdy->matprop;

  matprop->permeability_is_set = 1;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = PetscMemzero(matprop->K,sizeof(PetscReal)*dim2*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    f(&(tdy->X[dim*c]),&(matprop->K[dim2*(c-cStart)]));
  }
  ierr = PetscMemcpy(matprop->K0,matprop->K,sizeof(PetscReal)*dim2*(cEnd-cStart));
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPorosity(TDy tdy,SpatialFunction f) {
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  MaterialProp *matprop = tdy->matprop;

  matprop->porosity_is_set = 1;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(matprop->porosity,sizeof(PetscReal)*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) f(&(tdy->X[dim*c]),&(matprop->porosity[c-cStart]));
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetSoilSpecificHeat(TDy tdy,SpatialFunction f) {
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  MaterialProp *matprop = tdy->matprop;

  matprop->soil_specific_heat_is_set = 1;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(matprop->Cr,sizeof(PetscReal)*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) f(&(tdy->X[dim*c]),&(matprop->Cr[c-cStart]));
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetSoilDensity(TDy tdy,SpatialFunction f) {
  PetscInt dim,c,cStart,cEnd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscValidPointer(tdy,1);
  PetscValidPointer(f,2);
  MaterialProp *matprop = tdy->matprop;

  matprop->soil_density_is_set = 1;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMemzero(matprop->rhosoil,sizeof(PetscReal)*(cEnd-cStart)); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) f(&(tdy->X[dim*c]),&(matprop->rhosoil[c-cStart]));
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

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
