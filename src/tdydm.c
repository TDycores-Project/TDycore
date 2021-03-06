#include <private/tdydmimpl.h>
#include <tdytimers.h>

PetscErrorCode TDyCreateDM(DM *dm) {
  PetscErrorCode ierr;
  PetscInt       dim = 2;
  PetscReal      lower_bound_x = 0., lower_bound_y = lower_bound_x, 
                 lower_bound_z = lower_bound_x;
  PetscReal      upper_bound_x = 1., upper_bound_y = upper_bound_x, 
                 upper_bound_z = upper_bound_x;
  PetscInt       Nx = -999, Ny = -999, Nz = -999;
  PetscBool      found;
  char           mesh_filename[PETSC_MAX_PATH_LEN];
  char           string[512];
  char           word[32];
  
  mesh_filename[0] = '\0';

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
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  size_t len;
  ierr = PetscStrlen(mesh_filename, &len); CHKERRQ(ierr);
  if (!len){
    if (dim == 1) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,
              "ERROR: Only two or three dimensions currently supported.");
      int i = 0;
      if (Nx > 0) i++;
      if (Ny > 0) i++;
      if (Nz > 0) i++;
      if (i > 1) {
        strcpy(string,"ERROR: Number of grid cells must be defined in only one dimension for a 1D problem:");
        if (Nx > 0) sprintf(word," %d",Nx); else sprintf(word," ?");
        strcat(string,word);
        if (Ny > 0) sprintf(word," %d",Ny); else sprintf(word," ?");
        strcat(string,word);
        if (Nz > 0) sprintf(word," %d\n",Nz); else sprintf(word," ?");
        strcat(string,word);
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,string);
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
        strcpy(string,"ERROR: Number of grid cells must be defined in one (-N #) or two dimensions for a 2D problem:");
        if (Nx > 0) sprintf(word," %d",Nx); else sprintf(word," ?");
        strcat(string,word);
        if (Ny > 0) sprintf(word," %d",Ny); else sprintf(word," ?");
        strcat(string,word);
        if (Nz > 0) sprintf(word," %d\n",Nz); else sprintf(word," ?");
        strcat(string,word);
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,string);
      }
      if (Nx > 0 && Ny > 0) Nz = 1;
      else if (Nx > 0 && Nz > 0) Ny = 1;
      else if (Ny > 0 && Nz > 0) Nx = 1;
      else {Nx = 8; Ny = Nx; Nz = 1;}
    }
    else if (dim == 3) {
      if ((Nx < 0 && (Ny > 0 || Nz > 0)) || 
          (Nx > 0 && ((Ny > 0 && Nz < 0) || (Ny < 0 && Nz > 0)))) {
        strcpy(string,"ERROR: Number of grid cells must be defined in one (-N #) or three dimensions for a 3D problem:");
        if (Nx > 0) sprintf(word," %d",Nx); else sprintf(word," ?");
        strcat(string,word);
        if (Ny > 0) sprintf(word," %d",Ny); else sprintf(word," ?");
        strcat(string,word);
        if (Nz > 0) sprintf(word," %d\n",Nz); else sprintf(word," ?");
        strcat(string,word);
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,string);
      }
      if (Nx < 0) Nx = 8;
      if (Ny < 0) Ny = Nx;
      if (Nz < 0) Nz = Nx;
    }
    const PetscInt  faces[3] = {Nx,Ny,Nz};
    const PetscReal lower[3] = {lower_bound_x,lower_bound_y,lower_bound_z};
    const PetscReal upper[3] = {upper_bound_x,upper_bound_y,upper_bound_z};

    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces,
                               lower, upper, NULL, PETSC_TRUE, dm);
    CHKERRQ(ierr);
    PetscInt rank;
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)*dm),&rank);
    if (!rank) {
      int num_cell_in_set;
      num_cell_in_set = Nx * Ny * Nz;
      for (int c=0; c < num_cell_in_set; ++c)
        ierr = DMSetLabelValue(*dm, "Cell Sets", c, 1); CHKERRQ(ierr);
    }
  } else {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, mesh_filename, PETSC_TRUE,
                                dm); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyDistributeDM(DM *dm) {
  DM dmDist;
  PetscErrorCode ierr;

  /* Define 1 DOF on cell center of each cell */
  PetscInt dim;
  PetscFE fe;
  ierr = DMGetDimension(*dm, &dim); CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, "p_", -1, &fe); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "p");CHKERRQ(ierr);
  ierr = DMSetField(*dm, 0, NULL, (PetscObject)fe); CHKERRQ(ierr);
  ierr = DMCreateDS(*dm); CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  ierr = DMSetUseNatural(*dm, PETSC_TRUE); CHKERRQ(ierr);

  ierr = DMPlexDistribute(*dm, 1, NULL, &dmDist); CHKERRQ(ierr);
  if (dmDist) {
    DMDestroy(dm); CHKERRQ(ierr);
    *dm = dmDist;
  }
  ierr = DMSetFromOptions(*dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
