#include "tdycore.h"

/* ---------------------------------------------------------------- */
PetscErrorCode UpdateVertices(DM dm, PetscInt nx, PetscInt ny, PetscInt nz) {
  PetscReal Lx=1000.0, Ly=500.0, Lz=7.5;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value,dim;
  PetscReal x,y,z;
  PetscReal x_new,y_new,z_new;
  PetscErrorCode ierr;

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
    x = coords[offset  ];
    y = coords[offset+1];
    z = coords[offset+2];
    
    x_new = Lx*x;
    y_new = Ly*(1.5-x)*y;
    z_new = Lz*(0.5*(PetscCosReal(x*PETSC_PI) + 1.0) + z );

    coords[offset  ] = x_new;
    coords[offset+1] = y_new;
    coords[offset+2] = z_new;
  }
  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
int main(int argc, char **argv) {
  PetscInt nx=10, ny=1, nz=5;
  PetscInt successful_exit_code=0;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Curvilinear hillslope problem options",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nx","Number of elements in xdir","",nx,&nx,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ny","Number of elements in ydir","",ny,&ny,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nz","Number of elements in zdir","",nz,&nz,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  
  if (nx<=0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Bad value specified for -nx");
  if (ny<=0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Bad value specified for -ny");
  if (nz<=0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Bad value specified for -nz");

  DM dm, dmDist = NULL;
  PetscInt  faces[3];
  PetscReal lower[3];
  PetscReal upper[3];
  PetscInt dim = 3;

  faces[0] =  nx; faces[1] =   ny; faces[2] = nz;
  lower[0] = 0.0; lower[1] = -0.5; lower[2] = 0.0;
  upper[0] = 1.0; upper[1] =  0.5; upper[2] = 1.0;

  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
  ierr = UpdateVertices(dm,nx,ny,nz); CHKERRQ(ierr);

  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(successful_exit_code);

}
