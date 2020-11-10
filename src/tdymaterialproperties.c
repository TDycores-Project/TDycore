#include <private/tdycoreimpl.h>
#include <private/tdymaterialpropertiesimpl.h>
#include <tdytimers.h>

PetscErrorCode MaterialPropertiesCreate(PetscInt ndim, PetscInt ncells, MaterialProp **_matprop){

  PetscFunctionBegin;
  PetscErrorCode ierr;

  *_matprop = (MaterialProp *)malloc(sizeof(struct _MaterialProp));

  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->K    ); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->K0   ); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->Kappa); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*ndim*ndim*sizeof(PetscReal),&(*_matprop)->Kappa0); CHKERRQ(ierr);

  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_matprop)->porosity); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_matprop)->Cr      ); CHKERRQ(ierr);
  ierr = PetscMalloc(ncells*sizeof(PetscReal),&(*_matprop)->rhosoil    ); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
  Material properties set by PETSc operations
*/

PetscErrorCode TDySetPermeabilityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computepermeability = f;
  if (ctx) tdy->permeabilityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPorosityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
  if (f) tdy->ops->computeporosity = f;
  if (ctx) tdy->porosityctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetThermalConductivityFunction(TDy tdy, PetscErrorCode(*f)(TDy,PetscReal*,PetscReal*,void*),void *ctx) {
  PetscFunctionBegin;
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

PetscErrorCode TDySetBlockPermeabilityValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i,j;
  PetscInt dim,dim2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ni) PetscFunctionReturn(0);

  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  MaterialProp *matprop = tdy->matprop;

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

  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&i); CHKERRQ(ierr);
  ierr = DMGetDimension(tdy->dm,&dim2); CHKERRQ(ierr);
  dim2 *= dim2;  
  for(i=0;i<dim2;i++) matprop->K[dim2*(c-cStart)+i] = K[i];
  PetscFunctionReturn(0);
}

PetscErrorCode TDySetPorosityValuesLocal(TDy tdy, PetscInt ni, const PetscInt ix[], const PetscScalar y[]){

  PetscInt i;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  if (!ni) PetscFunctionReturn(0);

  MaterialProp *matprop = tdy->matprop;

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

void TDySoilDensityFunctionDefault(PetscReal *x, PetscReal *den) {
  *den = 2650.;
}

void TDySpecificSoilHeatFunctionDefault(PetscReal *x, PetscReal *cr) {
  *cr = 1000.;
}

PetscErrorCode TDyPermeabilityFunctionDefault(TDy tdy, PetscReal *x, PetscReal *K, void *ctx) {
  PetscErrorCode ierr;
  PetscInt dim;
  PetscFunctionBegin;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  for (int j=0; j<dim; j++) {
    for (int i=0; i<dim; i++) {
      K[j*dim+i] = 0.;
    }
  }
  for (int i=0; i<dim; i++)
    K[i*dim+i] = 1.e-12;
  return 0;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyThermalConductivityFunctionDefault(TDy tdy, double *x, double *K, void *ctx) {
  PetscErrorCode ierr;
  PetscInt dim;
  PetscFunctionBegin;
  ierr = DMGetDimension(tdy->dm,&dim); CHKERRQ(ierr);
  for (int j=0; j<dim; j++) {
    for (int i=0; i<dim; i++) {
      K[j*dim+i] = 0.;
    }
  }
  for (int i=0; i<dim; i++)
    K[i*dim+i] = 1.;
  return 0;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyPorosityFunctionDefault(TDy tdy, double *x, double *por, void *ctx) {
  PetscFunctionBegin;
  *por = 0.25;
  PetscFunctionReturn(0);
}
