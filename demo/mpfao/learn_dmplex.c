#include "tdycore.h"

int main(int argc, char **argv) {

  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char *)0,0);CHKERRQ(ierr);

  DM dm;
  PetscInt n = 5;
  const PetscInt num_cells_in_direction[3] = {n,n,n};
  const PetscReal lower_bound[3] = {0.,0.,0.};
  const PetscReal upper_bound[3] = {1.,1.,1.};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,3,PETSC_FALSE,
                             num_cells_in_direction,lower_bound,upper_bound,
                             NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = DMView(dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  printf("\nFor #-cells above, # refers to the index in the DMPlexGetDepthStratum, where for 3D finite volume: 0 = vertices, 1 = faces, 2 = edges, and 3 = cells. ");
  printf("Face sets refer to groups of faces on each side of the unit cube "); 
  printf("where the value is the local face id (?) and size is the number of faces with the id. "); 
  printf("Concidently, the box example has 6 sides just like each hexahedron. ");
  printf("Google DMPolytopeType for the enumeration of celltype values. ");
  printf("\n");

  PetscInt i, istart, iend;
  printf("\nOutput from DMPlexGetHeightStratum(dm,i,&istart,&iend)\n");
  printf("i    Title: istart -  iend ->  iend-istart\n");
  printf("---------------------------------------\n");
  i = 0;
  ierr = DMPlexGetHeightStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d    Cells:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);
  i = 1;
  ierr = DMPlexGetHeightStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d    Faces:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);
  i = 2;
  ierr = DMPlexGetHeightStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d    Edges:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);
  i = 3;
  ierr = DMPlexGetHeightStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d Vertices:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);

  printf("\nOutput from DMPlexGetDepthStratum(dm,i,&istart,&iend)\n");
  printf("i    Title: istart -  iend ->  iend-istart\n");
  printf("---------------------------------------\n");
  i = 3;
  ierr = DMPlexGetDepthStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d    Cells:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);
  i = 2;
  ierr = DMPlexGetDepthStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d    Faces:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);
  i = 1;
  ierr = DMPlexGetDepthStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d    Edges:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);
  i = 0;
  ierr = DMPlexGetDepthStratum(dm,i,&istart,&iend);CHKERRQ(ierr);
  printf("%d Vertices:  %5d - %5d -> %5d\n",i,istart,iend,iend-istart);

  printf("\n");
  printf("Number of cells should be: %d\n",n*n*n);
  printf("Number of faces should be: %d\n",n*n*(n+1)*3);
  printf("Number of edges should be: %d\n",n*(n+1)*(n+1)*3);
  printf("Number of verts should be: %d\n",(n+1)*(n+1)*(n+1));

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
}
