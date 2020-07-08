//#include <private/tdydmperturbationimpl.h>
#include <tdydmperturbation.h>

PetscErrorCode (*vertexperturbationfunction)(DM,PetscReal) = NULL;

PetscErrorCode SetVertexPerturbationFunction(PetscErrorCode(*f)(DM,PetscReal)) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (f) vertexperturbationfunction = f;
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbDMVertices(DM dm,PetscReal h) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (vertexperturbationfunction) {
    ierr = vertexperturbationfunction(dm,h);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbDMInteriorVertices(DM dm,PetscReal h) {
  PetscErrorCode ierr;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value;

  PetscFunctionBegin;

  ierr = DMGetLabelByNum(dm,2,&label);
  CHKERRQ(ierr); // this is the 'marker' label which marks boundary entities

  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords); CHKERRQ(ierr);

  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,v,&value); CHKERRQ(ierr);
    if(value==-1) {
      PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*
                    (h*0.471404); // h*sqrt(2)/3
      PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
      coords[offset  ] += r*PetscCosReal(t);
      coords[offset+1] += r*PetscSinReal(t);
    }
  }

  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PerturbVerticesRandom(DM dm,PetscReal h) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value,dim;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  /* this is the 'marker' label which marks boundary entities */
  ierr = DMGetLabelByNum(dm,2,&label); CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,v,&value); CHKERRQ(ierr);
    if(dim==2) {
      if(value==-1) {
        /* perturb randomly O(h*sqrt(2)/3) */
        PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*(h*0.471404);
        PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
        coords[offset  ] += r*PetscCosReal(t);
        coords[offset+1] += r*PetscSinReal(t);
      }
    } else {
      /* This is because 'marker' is broken in 3D */
      if(coords[offset] > 0 && coords[offset] < 1 &&
          coords[offset+1] > 0 && coords[offset+1] < 1 &&
          coords[offset+2] > 0 && coords[offset+2] < 1) {
        coords[offset+2] += (((PetscReal)rand())/((PetscReal)RAND_MAX)-0.5)*h*0.1; // <-- not removing yet but notice 1/10
      }
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PerturbVerticesSmooth(DM dm,PetscReal dummy) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,dim;
  PetscReal    x,y,z;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMGetLabelByNum(dm,2,&label); CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords); CHKERRQ(ierr);
  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    if(dim==2) {
      x = coords[offset]; y = coords[offset+1];
      coords[offset]   = x + 0.06*PetscSinReal(2.0*PETSC_PI*x)*PetscSinReal(2.0*PETSC_PI*y);
      coords[offset+1] = y - 0.05*PetscSinReal(2.0*PETSC_PI*x)*PetscSinReal(2.0*PETSC_PI*y);
    } else {
      x = coords[offset]; y = coords[offset+1]; z = coords[offset+2];
      coords[offset]   = x + 0.03*PetscSinReal(3*PETSC_PI*x)*PetscCosReal(3*PETSC_PI*y)*PetscCosReal(3*PETSC_PI*z);
      coords[offset+1] = y - 0.04*PetscCosReal(3*PETSC_PI*x)*PetscSinReal(3*PETSC_PI*y)*PetscCosReal(3*PETSC_PI*z);
      coords[offset+2] = z + 0.05*PetscCosReal(3*PETSC_PI*x)*PetscCosReal(3*PETSC_PI*y)*PetscSinReal(3*PETSC_PI*z);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
