#include <private/tdycoreimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <tdytimers.h>
#include <private/tdyoptions.h>

PetscErrorCode MaterialPropertiesCreate(PetscInt ndim, PetscInt ncells, MaterialProp **_matprop){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  *_matprop = (MaterialProp *)malloc(sizeof(MaterialProp));

  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->K    ); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->K0   ); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->Kappa); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->Kappa0); CHKERRQ(ierr);

  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_matprop)->porosity); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_matprop)->Cr      ); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_matprop)->rhosoil    ); CHKERRQ(ierr);

  (*_matprop)-> permeability_is_set = 0;
  (*_matprop)-> porosity_is_set = 0;
  (*_matprop)-> thermal_conductivity_is_set = 0;
  (*_matprop)-> soil_specific_heat_is_set = 0;
  (*_matprop)-> soil_density_is_set = 0;

  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPropertiesDestroy(MaterialProp *matprop){

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

//
// Are material properties set
//

/* -------------------------------------------------------------------------- */
/// Reports if permeability is set
///
/// @param [in] tdy    A TDy struct
/// @returns boolean   1 if permeability is set otherwise 0
PetscBool TDyIsPermeabilitySet(TDy tdy) {
  PetscFunctionReturn(tdy->matprop->permeability_is_set);
}

/* -------------------------------------------------------------------------- */
/// Reports if porosity is set
///
/// @param [in] tdy    A TDy struct
/// @returns boolean   1 if porosity is set otherwise 0
PetscBool TDyIsPorositySet(TDy tdy) {
  PetscFunctionReturn(tdy->matprop->porosity_is_set);
}

/* -------------------------------------------------------------------------- */
/// Reports if thermal conductivity is set
///
/// @param [in] tdy    A TDy struct
/// @returns boolean   1 if thermal conductivity is set otherwise 0
PetscBool TDyIsThermalConductivytSet(TDy tdy) {
  PetscFunctionReturn(tdy->matprop->thermal_conductivity_is_set);
}

/* -------------------------------------------------------------------------- */
/// Reports if soil specific heat is set
///
/// @param [in] tdy    A TDy struct
/// @returns boolean   1 if soil specific heat is set otherwise 0
PetscBool TDyIsSoilSpecificHeatSet(TDy tdy) {
  PetscFunctionReturn(tdy->matprop->soil_specific_heat_is_set);
}

/* -------------------------------------------------------------------------- */
/// Reports if soil density is set
///
/// @param [in] tdy    A TDy struct
/// @returns boolean   1 if soil density is set otherwise 0
PetscBool TDyIsSoilDensitySet(TDy tdy) {
  PetscFunctionReturn(tdy->matprop->soil_density_is_set);
}

/*
  Material properties set by PETSc operations
*/

PetscErrorCode TDySetPermeabilityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  tdy->matprop->permeability_is_set = 1;
  if (f) tdy->ops->computepermeability = f;
  if (ctx) tdy->permeabilityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPorosityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  tdy->matprop->porosity_is_set = 1;
  if (f) tdy->ops->computeporosity = f;
  if (ctx) tdy->porosityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetSoilDensityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  tdy->matprop->soil_density_is_set = 1;
  if (f) tdy->ops->computesoildensity = f;
  if (ctx) tdy->soildensityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetSoilSpecificHeatFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  tdy->matprop->soil_specific_heat_is_set = 1;
  if (f) tdy->ops->computesoilspecificheat = f;
  if (ctx) tdy->soilspecificheatctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetThermalConductivityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  tdy->matprop->thermal_conductivity_is_set = 1;
  if (f) tdy->ops->computethermalconductivity = f;
  if (ctx) tdy->thermalconductivityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetResidualSaturationFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeresidualsaturation = f;
  if (ctx) tdy->residualsaturationctx = ctx;
  PetscFunctionReturn(0);
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

/*
  Default function for material properties
*/

PetscErrorCode TDySoilDensityFunctionDefault(TDy tdy, PetscReal *x, PetscReal *den, void *ctx) {
  PetscFunctionBegin;
  TDyOptions *options = &tdy->options;

  *den = options->soil_density;

  PetscFunctionReturn(0);
}

PetscErrorCode TDySoilSpecificHeatFunctionDefault(TDy tdy, PetscReal *x, PetscReal *cr, void *ctx) {
  PetscFunctionBegin;
  TDyOptions *options = &tdy->options;

  *cr = options->soil_specific_heat;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyPermeabilityFunctionDefault(TDy tdy, PetscReal *x, PetscReal *K, void *ctx) {
  PetscFunctionBegin;
  TDyOptions *options = &tdy->options;
  PetscInt dim;
  PetscErrorCode ierr;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  for (int j=0; j<dim; j++) {
    for (int i=0; i<dim; i++) {
      K[j*dim+i] = 0.;
    }
  }
  for (int i=0; i<dim; i++)
    K[i*dim+i] = options->permeability;
  return 0;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyThermalConductivityFunctionDefault(TDy tdy, PetscReal *x, PetscReal *K, void *ctx) {
  PetscFunctionBegin;
  TDyOptions *options = &tdy->options;
  PetscErrorCode ierr;
  PetscInt dim;

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  for (int j=0; j<dim; j++) {
    for (int i=0; i<dim; i++) {
      K[j*dim+i] = 0.;
    }
  }
  for (int i=0; i<dim; i++)
    K[i*dim+i] = options->thermal_conductivity;
  return 0;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyPorosityFunctionDefault(TDy tdy, PetscReal *x, PetscReal *por, void *ctx) {
  PetscFunctionBegin;
  TDyOptions *options = &tdy->options;
  *por = options->porosity;
  PetscFunctionReturn(0);
}
