#include "tdycore.h"

/*
  BDM1 basis functions on [-1,1] with degrees of freedom chosen to
  match Wheeler2009. Indices map <-- local_vertex*dim + dir.

  2---3
  |   |
  0---1

 */
void HdivBasisQuad(PetscReal *x,PetscReal *B){
  B[0] = -0.25*x[0]*x[1] + 0.25*x[0] + 0.25*x[1] - 0.25;
  B[1] = -0.25*x[0]*x[1] + 0.25*x[0] + 0.25*x[1] - 0.25;
  B[2] = -0.25*x[0]*x[1] + 0.25*x[0] - 0.25*x[1] + 0.25;
  B[3] = +0.25*x[0]*x[1] - 0.25*x[0] + 0.25*x[1] - 0.25;
  B[4] = +0.25*x[0]*x[1] + 0.25*x[0] - 0.25*x[1] - 0.25;
  B[5] = -0.25*x[0]*x[1] - 0.25*x[0] + 0.25*x[1] + 0.25;
  B[6] = +0.25*x[0]*x[1] + 0.25*x[0] + 0.25*x[1] + 0.25;
  B[7] = +0.25*x[0]*x[1] + 0.25*x[0] + 0.25*x[1] + 0.25;
}

PetscErrorCode TDyMFELocalElementCompute(DM dm,TDy tdy)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

PetscErrorCode TDyMFEInitialize(DM dm,TDy tdy){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt pStart,pEnd,c,cStart,cEnd,f,fStart,fEnd;
  PetscSection sec;
  PetscInt d,dim,dofs_per_face = 1;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  
  /* Get plex limits */
  ierr = DMPlexGetChart        (dm,  &pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);

  /* Create H-div section */
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&sec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,2);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Pressure");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,1,"Velocity");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,1,1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd);CHKERRQ(ierr);
  
  /* Setup 1 dof per cell for field 0 */
  for(c=cStart;c<cEnd;c++){
    ierr = PetscSectionSetFieldDof(sec,c,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec,c  ,1); CHKERRQ(ierr);
  }

  /* Setup dofs_per_face considering quads and hexes only */
  for(d=0;d<(dim-1);d++) dofs_per_face *= 2;
  for(f=fStart;f<fEnd;f++){
    ierr = PetscSectionSetFieldDof(sec,f,1,dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec,f  ,dofs_per_face); CHKERRQ(ierr);
  }  
  ierr = PetscSectionSetUp(sec);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,sec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec);CHKERRQ(ierr);

  /* I am not sure what we want here, but this seems to be a
     conservative estimate on the sparsity we need. */
  ierr = DMPlexSetAdjacencyUseCone   (dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  /* */


  
  PetscFunctionReturn(0);
}

PetscErrorCode TDyMFEComputeSystem(DM dm,TDy tdy,Mat K,Vec F){
  PetscFunctionBegin;  
  PetscErrorCode ierr;

  /* A_ij = (Kappa^-1 u_i,v_j)
     Loop over cells, use full quadrature, needs to capture x**2 y**2? I think 3x3
       Loop over vertices twice, integrate each interaction
         Leads to a dense 8x8 matrix

     1 corner vs 1 corner
     u1(xq) = | u11 N1(xq) | 
              | u12 N2(xq) |
     v1(xq) = | v11 N1(xq) | 
              | v12 N2(xq) |

   ( | K11, K12 |  | u11 N1(xq)| )   | v11 N1(xq) |
   ( | K21, K22 |. | u12 N2(xq)| ) . | v12 N2(xq) |

     | K11 u11 N1(xq) + K12 u12 N2(xq) |   | v11 N1(xq) |
     | K21 u11 N1(xq) + K22 u12 N2(xq) | . | v12 N2(xq) |

     Each vertex-vertex interaction leads to a 2x2 matrix

     [ K11 N1(xq) N1(xq), K12 N2(xq) N1(xq) ]
     [ K21 N1(xq) N2(xq), K22 N2(xq) N2(xq) ] * wq

     Loop over cells
       Loop over quadrature
         Loop over vertices twice
	   Assemble in the 8x8 local matrix

     Need:
       map(face,vertex) --> global_flux_dof + orientation
       map(cell,local_vertex,direction) --> local_element_flux_dof, this is already defined
       map(cell,local_element_flux_dof) --> global_flux_dof
  */
  
  
  /* B_ij = <p, v_j.n> 
     Loop over cells
       Evaluate line integral, by loop around faces, 1 point quadrature is fine

     <p, v_1.n> = v11 + v12 = [1,1,0,0,0,0,0,0]
     <p, v_2.n> = v21 + v22 = [0,0,1,1,0,0,0,0]
     ...
     Assemble [1,1,1,1,1,1,1,1] into global_cell x global_faces modified by direction
     
     Don't actually need Hdiv basis functions here
  */
  
  
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  
  stack time traces and show seasons from gpp
  seasons across different variables make sense? Use gpp season across all variables?
  lai, from MODIS
  
 */
