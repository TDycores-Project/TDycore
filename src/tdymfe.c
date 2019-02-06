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
  PetscInt pStart,pEnd,c,cStart,cEnd,f,f_abs,fStart,fEnd,nfv,ncv,v,vStart,vEnd,mStart,mEnd,i,nlocal,closureSize,*closure;
  PetscSection sec;
  PetscInt d,dim,dofs_per_face = 1;
  PetscBool found;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  
  /* Get plex limits */
  ierr = DMPlexGetChart        (dm,  &pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
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
  ierr = DMGetDefaultGlobalSection(dm,&sec);CHKERRQ(ierr);
  
  /* I am not sure what we want here, but this seems to be a
     conservative estimate on the sparsity we need. */
  ierr = DMPlexSetAdjacencyUseCone   (dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  /* Build vmap and emap */
  ierr = TDyCreateCellVertexMap(dm,tdy,&(tdy->vmap));CHKERRQ(ierr);
  ierr = TDyCreateCellVertexDirFaceMap(dm,tdy,&(tdy->emap));CHKERRQ(ierr);

  /* Build map(face,local_vertex) --> vertex */
  nfv = TDyGetNumberOfFaceVertices(dm);
  ierr = PetscMalloc(nfv*(fEnd-fStart)*sizeof(PetscInt),&(tdy->fmap));CHKERRQ(ierr);
  for(f=fStart;f<fEnd;f++){
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    i = 0;
    for(c=0;c<closureSize*2;c+=2){
      if ((closure[c] < vStart) || (closure[c] >= vEnd)) continue;
      tdy->fmap[nfv*(f-fStart)+i] = closure[c];
      i += 1;
    }
    if(i != nfv){
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Irregular number of vertices per face found");
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);    
  }

  /* use vmap, emap, and fmap to build a LtoG map for local element
     assembly */
  ncv = TDyGetNumberOfCellVertices(dm);
  nlocal = dim*ncv + 1;
  ierr = PetscMalloc((cEnd-cStart)*nlocal*sizeof(PetscInt),&(tdy->LtoG));CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    DMPlexGetPointGlobal(dm,c,&mStart,&mEnd);CHKERRQ(ierr);
    tdy->LtoG[(c-cStart+1)*nlocal-1] = mStart;
    for(v=0;v<ncv;v++){
      for(d=0;d<dim;d++){
	f = tdy->emap[(c-cStart)*ncv*dim+v*dim+d]; /* which face is this local dof on? */
	f_abs = PetscAbsInt(f);
	ierr = DMPlexGetPointGlobal(dm,f_abs,&mStart,&mEnd);CHKERRQ(ierr);
	found = PETSC_FALSE;
	for(i=0;i<nfv;i++){
	  if(tdy->vmap[ncv*(c-cStart)+v] == tdy->fmap[nfv*(f_abs-fStart)+i]){
	    tdy->LtoG[(c-cStart)*nlocal + v*dim + d] = mStart + i;
	    found = PETSC_TRUE;
	  }
	}
	if(!found){
	  SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Could not find a face vertex for this cell");
	}
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyMFEComputeSystem(DM dm,TDy tdy,Mat K,Vec F){
  PetscFunctionBegin;  
  PetscErrorCode ierr;
  PetscInt dim,dim2,nlocal,pStart,pEnd,c,cStart,cEnd,q,nq,nv,vi,vj,di,dj,local_row,local_col,isbc;
  PetscScalar x[24],DF[72],DFinv[72],J[8],Kinv[9],Klocal[MAX_LOCAL_SIZE],Flocal[MAX_LOCAL_SIZE],f,basis_hdiv[24],p;
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
    const PetscInt *LtoG = &(tdy->LtoG[(c-cStart)*nlocal]);
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quadrature,x,DF,DFinv,J);CHKERRQ(ierr);
    ierr = PetscMemzero(Klocal,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(Flocal,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);

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
	Flocal[nlocal-1] += f*J[q]*quad_w[q];
      }
      
    } /* end quadrature */

    /* <p, v_j.n> */
    for(local_col=0;local_col<(nlocal-1);local_col++){
      Klocal[local_col *nlocal + (nlocal-1)] = 1;
      Klocal[(nlocal-1)*nlocal + local_col ] = 1;
    }

    /* <g, v_j.n> */
    if(tdy->dirichlet){
      /* loop over all possible v_j's for this cell, integrating with
	 Gauss-Lobotto */
      for(vi=0;vi<nv;vi++){
	for(di=0;di<dim;di++){
	  f = tdy->emap[(c-cStart)*nv*dim + vi*dim + di];
	  ierr = DMGetLabelValue(dm,"marker",f,&isbc);CHKERRQ(ierr);
	  if(isbc == 1){
	    local_row = vi*dim+di;
	    tdy->dirichlet(&(tdy->X[(tdy->vmap[(c-cStart)*nv+vi])*dim]),&p);
	    Flocal[local_row] += p; /* Need to think about this */
	  }
	}
      }
    }
    
    /* apply orientation flips */
    
    /* assembly */
    ierr = MatSetValues(K,nlocal,LtoG,nlocal,LtoG,Klocal,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(F,nlocal,LtoG,Flocal,INSERT_VALUES);CHKERRQ(ierr);

  } /* end cell */

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = PetscQuadratureDestroy(&quadrature);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
