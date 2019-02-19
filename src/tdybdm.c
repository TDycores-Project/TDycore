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

PetscErrorCode TDyBDMInitialize(TDy tdy){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt pStart,pEnd,c,cStart,cEnd,f,f_abs,fStart,fEnd,nfv,ncv,v,vStart,vEnd,mStart,mEnd,i,nlocal,closureSize,*closure;
  PetscSection sec;
  PetscInt d,dim,dofs_per_face = 1;
  PetscBool found;
  DM dm = tdy->dm;
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
  ierr = TDyCreateCellVertexMap(tdy,&(tdy->vmap));CHKERRQ(ierr);
  ierr = TDyCreateCellVertexDirFaceMap(tdy,&(tdy->emap));CHKERRQ(ierr);

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
  ierr = PetscMalloc((cEnd-cStart)*nlocal*sizeof(PetscInt),&(tdy->orient));CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexGetPointGlobal(dm,c,&mStart,&mEnd);CHKERRQ(ierr);
    tdy->LtoG[(c-cStart+1)*nlocal-1] = mStart;
    for(v=0;v<ncv;v++){
      for(d=0;d<dim;d++){
	f = tdy->emap[(c-cStart)*ncv*dim+v*dim+d]; /* which face is this local dof on? */
	f_abs = PetscAbsInt(f);
	ierr = DMPlexGetPointGlobal(dm,f_abs,&mStart,&mEnd);CHKERRQ(ierr);
	found = PETSC_FALSE;
	for(i=0;i<nfv;i++){
	  if(tdy->vmap[ncv*(c-cStart)+v] == tdy->fmap[nfv*(f_abs-fStart)+i]){
	    tdy->LtoG  [(c-cStart)*nlocal + v*dim + d] = mStart + i;
	    tdy->orient[(c-cStart)*nlocal + v*dim + d] = PetscSign(f);
	    found = PETSC_TRUE;
	  }
	}
	if(!found){
	  SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Could not find a face vertex for this cell");
	}
      }
    }
  }

  /* map(cell,dim,side) --> global_face */
  ierr = PetscMalloc((cEnd-cStart)*PetscPowInt(2,dim)*sizeof(PetscInt),&(tdy->faces));CHKERRQ(ierr);
  PetscInt s;
  for(c=cStart;c<cEnd;c++){
    for(d=0;d<dim;d++){
      for(s=0;s<2;s++){
	v = s*PetscPowInt(2,d);	
	tdy->faces[(c-cStart)*dim*2+d*2+s] = PetscAbsInt(tdy->emap[(c-cStart)*ncv*dim+v*dim+d]);
      }
    }
  }
  
  PetscFunctionReturn(0);
}

/* x:  dim  *nq = 2*9 = 18 
   DF: dim^2*nq = 4*9 = 36
   J:        nq =   9 = 9
*/
PetscErrorCode TDyBDMComputeSystem(TDy tdy,Mat K,Vec F){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt dim,dim2,nlocal,pStart,pEnd,c,cStart,cEnd,q,nq,nv,vi,vj,di,dj,local_row,local_col,isbc,f;
  PetscScalar x[24],DF[72],DFinv[72],J[9],Kinv[9],Klocal[MAX_LOCAL_SIZE],Flocal[MAX_LOCAL_SIZE],force,basis_hdiv[24],pressure;
  const PetscScalar *quad_x;
  const PetscScalar *quad_w;
  PetscQuadrature quadrature;
  DM dm = tdy->dm;
  
  /* Get domain constants */
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr); dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  nv = TDyGetNumberOfCellVertices(dm);

  /* Get quadrature */
  ierr = PetscDTGaussTensorQuadrature(dim,1,3,-1,+1,&quadrature);CHKERRQ(ierr);
  //ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&quadrature);CHKERRQ(ierr);
  //ierr = TDyQuadrature(quadrature,dim);CHKERRQ(ierr);
  
  ierr = PetscQuadratureGetData(quadrature,NULL,NULL,&nq,&quad_x,&quad_w);CHKERRQ(ierr);  
  nlocal = dim*nv + 1;
  
  for(c=cStart;c<cEnd;c++){
    
    /* Only assemble the cells that this processor owns */
    ierr = DMPlexGetPointGlobal(dm,c,&pStart,&pEnd);CHKERRQ(ierr);
    if (pStart < 0) continue;
    const PetscInt *LtoG = &(tdy->LtoG[(c-cStart)*nlocal]);
    const PetscInt *orient = &(tdy->orient[(c-cStart)*nlocal]);
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
	      Klocal[local_col*nlocal+local_row] += Kinv[dj*dim+di]*basis_hdiv[local_row]*basis_hdiv[local_col]*quad_w[q]*0.25;
	      
	    }
	  } /* end directions */
					 
	}
      } /* end vertices */

      /* Integrate forcing if present */
      if (tdy->forcing) {
	(*tdy->forcing)(&(x[q*dim]),&force);
	Flocal[nlocal-1] += force*J[q]*quad_w[q];
      }
      
    } /* end quadrature */

    /* <p, v_j.n> */
    for(local_col=0;local_col<(nlocal-1);local_col++){
      Klocal[local_col *nlocal + (nlocal-1)] = 0.5;
      Klocal[(nlocal-1)*nlocal + local_col ] = 0.5;
    }

    /* <g, v_j.n> */
    if(tdy->dirichlet){
      /* loop over all possible v_j's for this cell, integrating with
	 Gauss-Lobotto */
      for(vi=0;vi<nv;vi++){
	for(di=0;di<dim;di++){
	  f = PetscAbsInt(tdy->emap[(c-cStart)*nv*dim + vi*dim + di]);
	  ierr = DMGetLabelValue(dm,"marker",f,&isbc);CHKERRQ(ierr);
	  if(isbc == 1){
	    local_row = vi*dim+di;
	    tdy->dirichlet(&(tdy->X[(tdy->vmap[(c-cStart)*nv+vi])*dim]),&pressure);
	    Flocal[local_row] += 0.5*pressure; /* Need to think about this */
	  }
	}
      }
    }
    
    /* apply orientation flips */
    for(vi=0;vi<nlocal-1;vi++){
      Flocal[vi] *= (PetscScalar)orient[vi];
      for(vj=0;vj<nlocal-1;vj++){
	Klocal[vj*nlocal+vi] *= (PetscScalar)(orient[vi]*orient[vj]);
      }
      Klocal[(nlocal-1)*nlocal+vi] *= (PetscScalar)orient[vi];
      Klocal[vi*nlocal+nlocal-1  ] *= (PetscScalar)orient[vi];
    }
    
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

PetscReal TDyBDMPressureNorm(TDy tdy,Vec U)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscSection sec;
  PetscInt c,cStart,cEnd,offset,dim,gref,junk;
  PetscReal p,*u,norm,norm_sum;
  DM dm = tdy->dm;
  if(!(tdy->dirichlet)){
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Must set the pressure function with TDySetDirichletFunction");
  }
  norm = 0;
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm,&sec);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk);CHKERRQ(ierr);
    if(gref<0) continue;
    ierr = PetscSectionGetOffset(sec,c,&offset);CHKERRQ(ierr);
    tdy->dirichlet(&(tdy->X[c*dim]),&p);
    norm += tdy->V[c]*PetscSqr(u[offset]-p);
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)U));CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(norm_sum);
}

PetscReal TDyBDMVelocityNorm(TDy tdy,Vec U)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt nlocal,ncv,dim,c,cStart,cEnd,s,vv,d,dd,f,q,nq,i,j,nq1d=3;
  PetscReal xq[3],vel[3],vel0[3],x[27],J[9],N[24],norm=0,norm_sum=0;
  const PetscScalar *quad_x,*quad_w;
  PetscQuadrature quadrature;
  PetscScalar *u,face_error,ve,va;
  DM dm = tdy->dm;
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ncv = TDyGetNumberOfCellVertices(dm);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscDTGaussTensorQuadrature(dim-1,1,nq1d,-1,+1,&quadrature);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadrature,NULL,NULL,&nq,&quad_x,&quad_w);CHKERRQ(ierr); 
  nlocal = ncv*dim + 1;
  for(c=cStart;c<cEnd;c++){ /* loop cells */
    
    for(d=0;d<dim;d++){
      for(s=0;s<2;s++){ /* loop faces */	
	f = tdy->faces[(c-cStart)*PetscPowInt(2,dim)+2*d+s];
	
	/* integrate over face */
	face_error = 0;
	ierr = DMPlexComputeCellGeometryFEM(dm,f,quadrature,x,NULL,NULL,J);CHKERRQ(ierr);
	for(q=0;q<nq;q++){
	  
	  /* extend the dim-1 quadrature point to dim */
	  j = 0;
	  for(i=0;i<dim;i++){
	    if(i == d){
	      xq[i] = PetscPowInt(-1,s+1);
	    }else{
	      xq[i] = quad_x[q*dim+j];
	      j += 1;
	    }
	  }
	  
	  /* interpolate the normal component of the velocity */
	  HdivBasisQuad(xq,N);
	  vel[0] = 0; vel[1] = 0; vel[2] = 0;
	  for(vv=0;vv<ncv;vv++)
	    for(dd=0;dd<dim;dd++)
	      vel[dd] += N[vv*dim+dd]*u[tdy->LtoG[(c-cStart)*nlocal+vv*dim+dd]];
	  tdy->flux(x,vel0);
	  ve = TDyADotB(vel0,&(tdy->N[f*dim]),dim);
	  va = TDyADotB(vel ,&(tdy->N[f*dim]),dim);
	  
	  /* error norm using (3.40) of Wheeler2012 */
	  face_error += PetscSqr(ve-va)*quad_w[q]*J[q];
	}
	norm += face_error*tdy->V[c]/tdy->V[f];
      }
    }
    
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  PetscFunctionReturn(norm_sum);
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quadrature);CHKERRQ(ierr);
  PetscFunctionReturn(norm_sum);  
}

PetscReal TDyBDMDivergenceNorm(TDy tdy,Vec U)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
