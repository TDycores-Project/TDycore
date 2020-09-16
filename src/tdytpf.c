#include <private/tdycoreimpl.h>
#include <tdytimers.h>

PETSC_STATIC_INLINE void Waxpy(PetscInt dim, PetscScalar a,
                               const PetscScalar *x, const PetscScalar *y, PetscScalar *w) {PetscInt d; for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];}
PETSC_STATIC_INLINE PetscScalar Dot(PetscInt dim, const PetscScalar *x,
                                    const PetscScalar *y) {PetscScalar sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*y[d]; return sum;}
PETSC_STATIC_INLINE PetscReal Norm(PetscInt dim, const PetscScalar *x) {return PetscSqrtReal(PetscAbsScalar(Dot(dim,x,x)));}

PetscErrorCode TDyTPFInitialize(TDy tdy) {
  PetscErrorCode ierr;
  MPI_Comm comm;
  PetscInt dim,c,cStart,cEnd,pStart,pEnd;
  PetscSection sec;
  DM dm = tdy->dm;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);

  /* Setup the section, 1 dof per cell */
  ierr = PetscSectionCreate(comm,&sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Pressure"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1); CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    ierr = PetscSectionSetFieldDof(sec,c,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,c,1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec); CHKERRQ(ierr);
  ierr = DMSetSection(dm,sec); CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view"); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE); CHKERRQ(ierr);
  //ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_FALSE); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

PetscErrorCode TDyTPFComputeSystem(TDy tdy,Mat K,Vec F) {
  PetscErrorCode ierr;
  PetscInt dim,dim2,f,fStart,fEnd,c,cStart,cEnd,row,col,junk,ss;
  PetscReal pnt2pnt[3],dist,Ki,p,force;
  DM dm = tdy->dm;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  if(!tdy->allow_unsuitable_mesh) {
    ierr = TDyTPFCheckMeshSuitability(tdy); CHKERRQ(ierr);
  }
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  for(f=fStart; f<fEnd; f++) {

    const PetscInt *supp;
    ierr = DMPlexGetSupportSize(dm,f,&ss  ); CHKERRQ(ierr);
    ierr = DMPlexGetSupport    (dm,f,&supp); CHKERRQ(ierr);

    // wrt
    // v = -Ki (pd-p0)/dist
    // 00 Ki/dist
    // 0 -Ki pd / dist
    if(ss==1 && tdy->ops->computedirichletvalue) {
      ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]),&p, tdy->dirichletvaluectx);CHKERRQ(ierr);
      if (fabs(p+999.)<1.e-40) continue;
      ierr = DMPlexGetPointGlobal(dm,supp[0],&row,&junk); CHKERRQ(ierr);
      Waxpy(dim,-1,&(tdy->X[supp[0]*dim]),&(tdy->X[f*dim]),pnt2pnt);
      dist = Norm(dim,pnt2pnt);
      Ki = tdy->K[(supp[0]-cStart)*dim2];
      ierr = MatSetValue(K,row,row,Ki/dist*tdy->V[f]/tdy->V[supp[0]],ADD_VALUES);
      CHKERRQ(ierr);
      ierr = VecSetValue(F,row,p*Ki/dist*tdy->V[f]/tdy->V[supp[0]],ADD_VALUES);
      CHKERRQ(ierr);
      continue;
    }

    Waxpy(dim,-1,&(tdy->X[supp[0]*dim]),&(tdy->X[supp[1]*dim]),pnt2pnt);
    dist = Norm(dim,pnt2pnt);
    Ki = 0.5*(tdy->K[(supp[0]-cStart)*dim2]+tdy->K[(supp[1]-cStart)*dim2]);

    //wrt 0
    // v = -Ki (p1-p0)/dist
    // 00  Ki/dist
    // 01 -Ki/dist

    //wrt 1
    // v = -Ki (p0-p1)/dist
    // 11  Ki/dist
    // 10 -Ki/dist

    ierr = DMPlexGetPointGlobal(dm,supp[0],&row,&junk); CHKERRQ(ierr);
    ierr = DMPlexGetPointGlobal(dm,supp[1],&col,&junk); CHKERRQ(ierr);
    ierr = MatSetValue(K,row,row, Ki/dist*tdy->V[f]/tdy->V[supp[0]],ADD_VALUES);
    CHKERRQ(ierr);
    ierr = MatSetValue(K,row,col,-Ki/dist*tdy->V[f]/tdy->V[supp[0]],ADD_VALUES);
    CHKERRQ(ierr);
    ierr = MatSetValue(K,col,col, Ki/dist*tdy->V[f]/tdy->V[supp[1]],ADD_VALUES);
    CHKERRQ(ierr);
    ierr = MatSetValue(K,col,row,-Ki/dist*tdy->V[f]/tdy->V[supp[1]],ADD_VALUES);
    CHKERRQ(ierr);
  }

  if(tdy->ops->computeforcing) {
    for(c=cStart; c<cEnd; c++) {
      ierr = DMPlexGetPointGlobal(dm,c,&row,&junk); CHKERRQ(ierr);
      if(row < 0) continue;
      ierr = (*tdy->ops->computeforcing)(tdy,&(tdy->X[c*dim]),&force,tdy->forcingctx);CHKERRQ(ierr);;
      ierr = VecSetValue(F,row,force,ADD_VALUES); CHKERRQ(ierr);
    }
  }

  // max = 3.63988
  // min = 3.15550

  ierr = VecAssemblyBegin(F); CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/*
  norm = sqrt( sum_i^n  V_i * ( p(X_i) - P_i )^2 )

  where n is the number of cells.
 */
PetscReal TDyTPFPressureNorm(TDy tdy,Vec U) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscSection sec;
  PetscInt c,cStart,cEnd,offset,dim,gref,junk;
  PetscReal p,*u,norm,norm_sum;
  DM dm = tdy->dm;
  if(!(tdy->ops->computedirichletvalue)) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must set the pressure function with TDySetDirichletValueFunction");
  }
  norm = 0;
  ierr = VecGetArray(U,&u); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = DMGetSection(dm,&sec); CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetPointGlobal(dm,c,&gref,&junk); CHKERRQ(ierr);
    if(gref<0) continue;
    ierr = PetscSectionGetOffset(sec,c,&offset); CHKERRQ(ierr);
    ierr = (*tdy->ops->computedirichletvalue)(tdy,&(tdy->X[c*dim]),&p,tdy->dirichletvaluectx);CHKERRQ(ierr);
    norm += tdy->V[c]*PetscSqr(u[offset]-p);
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)U)); CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  ierr = VecRestoreArray(U,&u); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(norm_sum);
}

PetscReal TDyTPFVelocityNorm(TDy tdy,Vec U) {
  PetscErrorCode ierr;
  PetscInt dim,dim2,i,f,fStart,fEnd,c,cStart,cEnd,row,junk;
  PetscReal pnt2pnt[3],dist,Ki,p,vel[3],va,ve,*u,sign,face_error,norm,norm_sum;
  DM dm = tdy->dm;
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  dim2 = dim*dim;
  ierr = VecGetArray(U,&u); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  face_error = 0;
  norm = 0;

  for(c=cStart; c<cEnd; c++) {

    PetscInt cs;
    const PetscInt *cone;
    ierr = DMPlexGetConeSize(dm,c,&cs  ); CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,c,&cone); CHKERRQ(ierr);
    for(i=0; i<cs; i++) {
      f = cone[i];

      PetscInt ss;
      const PetscInt *supp;
      ierr = DMPlexGetSupportSize(dm,f,&ss  ); CHKERRQ(ierr);
      ierr = DMPlexGetSupport    (dm,f,&supp); CHKERRQ(ierr);

      if(ss==1 && tdy->ops->computedirichletvalue) {
        ierr = DMPlexGetPointGlobal(dm,supp[0],&row,&junk); CHKERRQ(ierr);
        Waxpy(dim,-1,&(tdy->X[supp[0]*dim]),&(tdy->X[f*dim]),pnt2pnt);
        dist = Norm(dim,pnt2pnt);
        Ki = tdy->K[(supp[0]-cStart)*dim2];
        ierr = (*tdy->ops->computedirichletvalue)(tdy, &(tdy->X[f*dim]),&p, tdy->dirichletvaluectx);CHKERRQ(ierr);
        sign = PetscSign(TDyADotBMinusC(&(tdy->N[dim*f]),&(tdy->X[dim*f]),
                                        &(tdy->X[dim*supp[0]]),dim));
        va = sign* -Ki*(p-u[(supp[0]-cStart)])/dist;
        ierr = (*tdy->ops->computedirichletflux)(tdy,&(tdy->X[f*dim]),&(vel[0]),tdy->dirichletfluxctx);CHKERRQ(ierr);
        ve = TDyADotB(vel,&(tdy->N[dim*f]),dim);
        face_error = PetscSqr((va-ve)/tdy->V[f]);
      } else {
        Waxpy(dim,-1,&(tdy->X[supp[0]*dim]),&(tdy->X[supp[1]*dim]),pnt2pnt);
        dist = Norm(dim,pnt2pnt);
        Ki = 0.5*(tdy->K[(supp[0]-cStart)*dim2]+tdy->K[(supp[1]-cStart)*dim2]);
        sign = PetscSign(TDyADotBMinusC(&(tdy->N[dim*f]),&(tdy->X[dim*f]),
                                        &(tdy->X[dim*c]),dim));
        va = sign* -Ki*(u[(supp[1]-cStart)]-u[(supp[0]-cStart)])/dist *tdy->V[f];
        if(c==supp[1]) va *= -1;
        ierr = (*tdy->ops->computedirichletflux)(tdy,&(tdy->X[f*dim]),&(vel[0]),tdy->dirichletfluxctx);CHKERRQ(ierr);
        ve = TDyADotB(vel,&(tdy->N[dim*f]),dim)*tdy->V[f];
        face_error = PetscSqr((va-ve)/tdy->V[f]);
      }
      norm += face_error*tdy->V[c];
    }
  }
  ierr = MPI_Allreduce(&norm,&norm_sum,1,MPIU_REAL,MPI_SUM,
                       PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);
  norm_sum = PetscSqrtReal(norm_sum);
  ierr = VecRestoreArray(U,&u); CHKERRQ(ierr);
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(norm_sum);
}

PetscErrorCode TDyTPFCheckMeshSuitability(TDy tdy) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  PetscInt dim,f,fStart,fEnd;
  PetscReal diff,dist,pnt2pnt[3];
  DM dm = tdy->dm;
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  for(f=fStart; f<fEnd; f++) {
    PetscInt ss;
    const PetscInt *supp;
    ierr = DMPlexGetSupportSize(dm,f,&ss  ); CHKERRQ(ierr);
    ierr = DMPlexGetSupport    (dm,f,&supp); CHKERRQ(ierr);
    if(ss==1) continue;
    Waxpy(dim,-1,&(tdy->X[supp[1]*dim]),&(tdy->X[supp[0]*dim]),pnt2pnt);
    dist = Norm(dim,pnt2pnt);
    diff = PetscAbsReal(TDyADotB(&(tdy->N[f*dim]),pnt2pnt,dim)/dist);
    if(PetscAbsReal(diff-1) > 10*PETSC_MACHINE_EPSILON) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP,
              "Mesh is unsuitable for a consistent two point flux method. To force rerun with -tdy_tpf_allow_unsuitable_mesh");
    }
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}
