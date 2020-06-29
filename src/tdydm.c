#include <private/tdydmimpl.h>

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

  /*
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, "coordinates.bin", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  VecView(coordinates, viewer); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  */

  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreateDM(DM *_dm) {
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       dim = 2;
  PetscInt       N = 8;
  PetscBool      perturb;
  char           mesh_filename[PETSC_MAX_PATH_LEN];

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"DM Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Problem dimension","",dim,&dim,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","Number of elements in 1D","",N,&N,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh_filename", "The mesh file", "",
                            mesh_filename, mesh_filename, PETSC_MAX_PATH_LEN,
                            NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-perturb","Perturb interior vertices","",perturb,
                          &perturb,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  size_t len;
  ierr = PetscStrlen(mesh_filename, &len); CHKERRQ(ierr);
  if (!len){
    const PetscInt  faces[3] = {N,N,N  };
    const PetscReal lower[3] = {0.0,0.0,0.0};
    const PetscReal upper[3] = {1.0,1.0,1.0};

    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces,
                               lower, upper, NULL, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
    if (perturb) {
      ierr = PerturbDMInteriorVertices(dm,1./N); CHKERRQ(ierr);
    } else {
      ierr = PerturbDMInteriorVertices(dm,0.); CHKERRQ(ierr);
    }
  } else {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, mesh_filename, PETSC_TRUE,
                                &dm); CHKERRQ(ierr);
  }

  DM dmDist;
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
  *_dm = dm;

  PetscFunctionReturn(0);
}

