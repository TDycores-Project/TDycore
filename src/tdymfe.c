#include "tdycore.h"

/* (dim*vertices_per_cell+1)^2 */
#define MAX_LOCAL_SIZE 625

/*
  BDM1 basis functions on [-1,1] with degrees of freedom chosen to
  match Wheeler2009. Indices map <-- local_vertex*dim + dir.

  2---3
  |   |
  0---1

 */
void HdivBasisQuad(const PetscReal *x,PetscReal *B){
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



  
  PetscFunctionReturn(0);
}

PetscErrorCode TDyMFEComputeSystem(DM dm,TDy tdy,Mat K,Vec F){
  PetscFunctionBegin;  
  PetscErrorCode ierr;
  PetscInt dim,dim2,nlocal,pStart,pEnd,c,cStart,cEnd,q,nq,nv,vi,vj,di,dj,local_row,local_col,LGmap[25];
  PetscScalar x[24],DF[72],DFinv[72],J[8],Kinv[9],Klocal[MAX_LOCAL_SIZE],Flocal,f,basis_hdiv[24];
  const PetscScalar *quad_x;
  const PetscScalar *quad_w;
  PetscQuadrature quadrature;

  /* Get domain constants */
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr); dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  nv = TDyGetNumberOfCellVertices(dm);

  /* Get quadrature */
  ierr = PetscDTGaussTensorQuadrature(dim,1,3,-1,+1,&quadrature);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadrature,NULL,NULL,&nq,&quad_x,&quad_w);CHKERRQ(ierr);  
  nlocal = dim*nv + 1;
  
  for(c=cStart;c<cEnd;c++){
    
    /* Only assemble the cells that this processor owns */
    ierr = DMPlexGetPointGlobal(dm,c,&pStart,&pEnd);CHKERRQ(ierr);
    if (pStart < 0) continue;
    LGmap[nlocal-1] = pStart;
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quadrature,x,DF,DFinv,J);CHKERRQ(ierr);
    ierr = PetscMemzero(Klocal,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    Flocal = 0;
    
    /* Integrate (Kappa^-1 u_i, v_j) */
    for(q=0;q<nq;q++){
      
      /* Compute (J DF^-1 K DF^-T )^-1 */
      ierr = Pullback(&(tdy->K[dim2*(c-cStart)]),&DFinv[dim2*q],Kinv,J[q],dim);CHKERRQ(ierr);

      /* Evaluate the H-div basis */
      HdivBasisQuad(&(quad_x[dim*q]),basis_hdiv);

      /* Double loop over local vertices */
      for(vi=0;vi<nv;vi++){
	for(vj=0;vj<nv;vj++){

	  /* Double loop over directions */
	  for(di=0;di<dim;di++){
	    local_row = vi*dim+di;
	    for(dj=0;dj<dim;dj++){
	      local_col = vj*dim+dj;

	      /* (K u, v) */
	      Klocal[local_col*nlocal+local_row] += Kinv[dj*dim+di]*basis_hdiv[local_row]*basis_hdiv[local_col]*quad_w[q];
	      
	    }
	  } /* end directions */
					 
	}
      } /* end vertices */

      /* Integrate forcing if present */
      if (tdy->forcing) {
	(*tdy->forcing)(&(x[q*dim]),&f);
	Flocal += f*J[q]*quad_w[q];
      }
      
    } /* end quadrature */

    /* <p, v_j.n> */
    for(local_col=0;local_col<(nlocal-1);local_col++){
      Klocal[local_col *nlocal + (nlocal-1)] = 1;
      Klocal[(nlocal-1)*nlocal + local_col ] = 1;
    }

    /* [cell,local_vertex,direction] --> dof and orientation */
    
    ierr = VecSetValue(F,LGmap[nlocal-1],Flocal,INSERT_VALUES);CHKERRQ(ierr);

  } /* end cell */

  
  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = PetscQuadratureDestroy(&quadrature);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
