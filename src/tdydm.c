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

  PetscFunctionReturn(0);
}

PetscErrorCode TDyCreateDM(DM *_dm) {
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       dim = 2;
  PetscReal      lower_bound_x = 0., lower_bound_y = lower_bound_x, 
                 lower_bound_z = lower_bound_x;
  PetscReal      upper_bound_x = 1., upper_bound_y = upper_bound_x, 
                 upper_bound_z = upper_bound_x;
  PetscInt       Nx = -999, Ny = -999, Nz = -999;
  PetscBool      found;
  PetscBool      perturb;
  char           mesh_filename[PETSC_MAX_PATH_LEN];

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"DM Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Problem dimension","",dim,&dim,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","Number of elements in 1D","",Nx,&Nx,&found);
                         CHKERRQ(ierr);
  if (found) {Ny = Nx; Nz = Nx;}
  ierr = PetscOptionsInt("-Nx","Number of elements in X","",Nx,&Nx,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny","Number of elements in Y","",Ny,&Ny,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nz","Number of elements in Z","",Nz,&Nz,NULL);
                         CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lower_bound","Lower bound","",lower_bound_x,
                          &lower_bound_x,&found); CHKERRQ(ierr);
  if (found) {lower_bound_y = lower_bound_x; lower_bound_z = lower_bound_x;}
  ierr = PetscOptionsReal("-lower_bound_x","Lower bound in X","",lower_bound_x,
                          &lower_bound_x,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lower_bound_y","Lower bound in Y","",lower_bound_y,
                          &lower_bound_y,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lower_bound_z","Lower bound in Z","",lower_bound_z,
                          &lower_bound_z,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-upper_bound","Upper bound","",upper_bound_x,
                          &upper_bound_x,&found); CHKERRQ(ierr);
  if (found) {upper_bound_y = upper_bound_x; upper_bound_z = upper_bound_x;}
  ierr = PetscOptionsReal("-upper_bound_x","Upper bound in X","",upper_bound_x,
                          &upper_bound_x,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-upper_bound_y","Upper bound in Y","",upper_bound_y,
                          &upper_bound_y,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-upper_bound_z","Upper bound in Z","",upper_bound_z,
                          &upper_bound_z,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh_filename", "The mesh file", "",
                            mesh_filename, mesh_filename, PETSC_MAX_PATH_LEN,
                            NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-perturb","Perturb interior vertices","",perturb,
                          &perturb,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  size_t len;
  ierr = PetscStrlen(mesh_filename, &len); CHKERRQ(ierr);
  if (!len){
    if (dim == 1) {
      printf("ERROR: Only two or three dimensions currently supported.\n");
      exit(0);
      int i = 0;
      if (Nx > 0) i++;
      if (Ny > 0) i++;
      if (Nz > 0) i++;
      if (i > 1) {
        printf("ERROR: Number of grid cells must be defined in only one ");
        printf("dimension for a 1D problem:");
        if (Nx > 0) printf(" %d",Nx); else printf(" ?");
        if (Ny > 0) printf(" %d",Ny); else printf(" ?");
        if (Nz > 0) printf(" %d\n",Nz); else printf(" ?\n");
        exit(0);
      }
      if (Nx > 0) {Ny = 1; Nz = 1;}
      else if (Ny > 0) {Nx = 1; Nz = 1;}
      else if (Nz > 0) {Nx = 1; Ny = 1;}
      else {Nx = 8; Ny = 1; Nz = 1;}
    } else if (dim == 2) {
      if (!((Nx > 0 && Ny > 0) || 
            (Nx > 0 && Nz > 0) || 
            (Ny > 0 && Nz > 0) ||
            (Nx > 0 && Ny < 0 && Nz < 0) ||
            (Nx < 0 && Ny < 0 && Nz < 0))) {
        printf("ERROR: Number of grid cells must be defined in one (-N #) ");
        printf("or two dimensions for a 2D problem:");
        if (Nx > 0) printf(" %d",Nx); else printf(" ?");
        if (Ny > 0) printf(" %d",Ny); else printf(" ?");
        if (Nz > 0) printf(" %d\n",Nz); else printf(" ?\n");
        exit(0);
      }
      if (Nx > 0 && Ny > 0) Nz = 1;
      else if (Nx > 0 && Nz > 0) Ny = 1;
      else if (Ny > 0 && Nz > 0) Nx = 1;
      else {Nx = 8; Ny = Nx; Nz = 1;}
    }
    else if (dim == 3) {
      if ((Nx < 0 && (Ny > 0 || Nz > 0)) || 
          (Nx > 0 && ((Ny > 0 && Nz < 0) || (Ny < 0 && Nz > 0)))) {
        printf("ERROR: Number of grid cells must be defined in one (-N #) ");
        printf("or three dimensions for a 3D problem:");
        if (Nx > 0) printf(" %d",Nx); else printf(" ?");
        if (Ny > 0) printf(" %d",Ny); else printf(" ?");
        if (Nz > 0) printf(" %d\n",Nz); else printf(" ?\n");
        exit(0);
      }
      if (Nx < 0) Nx = 8;
      if (Ny < 0) Ny = Nx;
      if (Nz < 0) Nz = Nx;
    }
    const PetscInt  faces[3] = {Nx,Ny,Nz};
    const PetscReal lower[3] = {lower_bound_x,lower_bound_y,lower_bound_z};
    const PetscReal upper[3] = {upper_bound_x,upper_bound_y,upper_bound_z};
    //printf("%d %d %d\n",faces[0],faces[1],faces[2]);
    //printf("%f %f %f\n",lower[0],lower[1],lower[2]);
    //printf("%f %f %f\n",upper[0],upper[1],upper[2]);

    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces,
                               lower, upper, NULL, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
    if (perturb) {
      ierr = PerturbDMInteriorVertices(dm,1./Nx); CHKERRQ(ierr);
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

