#include "private/tdyfeimpl.h"
#include <tdytimers.h>

const char *const TDyQuadratureTypes[] = {
  "LUMPED",
  "FULL",
  /* */
  "TDyQuadratureType","TDY_QUAD_",NULL
};

/* Check if the image of the quadrature point is coincident with
   the vertex, if so we create a map:

   map(cell,local_cell_vertex) --> vertex

   Allocates memory inside routine, user must free.
*/
PetscErrorCode CreateCellVertexMap(DM dm, PetscInt nv, PetscReal *X, PetscInt **map) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt dim,i,v,vStart,vEnd,c,cStart,cEnd,closureSize,*closure;
  PetscQuadrature quad;
  PetscReal x[24],DF[72],DFinv[72],J[8];
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&quad); CHKERRQ(ierr);
  ierr = SetQuadrature(quad,dim); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(nv*(cEnd-cStart)*sizeof(PetscInt),map); CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<nv*(cEnd-cStart); c++) { (*map)[c] = -1; }
#endif
  for(c=cStart; c<cEnd; c++) {
    ierr = DMPlexComputeCellGeometryFEM(dm,c,quad,x,DF,DFinv,J); CHKERRQ(ierr);
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    for(v=0; v<nv; v++) {
      for (i=0; i<closureSize*2; i+=2) {
        if ((closure[i] >= vStart) && (closure[i] < vEnd)) {
          if (TDyL1norm(&(x[v*dim]),&(X[closure[i]*dim]),dim) > 1e-12) continue;
          (*map)[c*nv+v] = closure[i];
          break;
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<nv*(cEnd-cStart); c++) {
    if((*map)[c]<0) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(cell,local_vertex) -> vertex");
    }
  }
#endif
  ierr = PetscQuadratureDestroy(&quad); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create a map:

   map(cell,local_cell_vertex,direction) --> face

   To do this, I loop over the vertices of this cell and find
   connected faces. Then I use the local ordering of the vertices to
   determine where the normal of this face points. Finally I check if
   the normal points into the cell. If so, then the index is given a
   negative as a flag later in the assembly process. Since the Hasse
   diagram always begins with cells, there isn't a conflict with 0
   being a possible point.
*/
PetscErrorCode CreateCellVertexDirFaceMap(DM dm, PetscInt nv, PetscReal *X,
                                          PetscReal *N, PetscInt *vmap,
                                          PetscInt **map) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt d,dim,i,f,fStart,fEnd,v,q,c,cStart,cEnd,closureSize,*closure,
           fclosureSize,*fclosure,local_dirs[24];
  if(!vmap) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
            "Must first create TDyCreateCellVertexMap to set vmap");
  }
  ierr = DMGetDimension(dm,&dim); CHKERRQ(ierr);
  if(dim == 2) {
    local_dirs[0] = 2; local_dirs[1] = 1;
    local_dirs[2] = 3; local_dirs[3] = 0;
    local_dirs[4] = 0; local_dirs[5] = 3;
    local_dirs[6] = 1; local_dirs[7] = 2;
  } else if(dim == 3) {
    local_dirs[0]  = 6; local_dirs[1]  = 5; local_dirs[2]  = 3;
    local_dirs[3]  = 7; local_dirs[4]  = 4; local_dirs[5]  = 2;
    local_dirs[6]  = 4; local_dirs[7]  = 7; local_dirs[8]  = 1;
    local_dirs[9]  = 5; local_dirs[10] = 6; local_dirs[11] = 0;
    local_dirs[12] = 2; local_dirs[13] = 1; local_dirs[14] = 7;
    local_dirs[15] = 3; local_dirs[16] = 0; local_dirs[17] = 6;
    local_dirs[18] = 0; local_dirs[19] = 3; local_dirs[20] = 5;
    local_dirs[21] = 1; local_dirs[22] = 2; local_dirs[23] = 4;
  }
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  ierr = PetscMalloc(dim*nv*(cEnd-cStart)*sizeof(PetscInt),map); CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<dim*nv*(cEnd-cStart); c++) { (*map)[c] = 0; }
#endif
  for(c=cStart; c<cEnd; c++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    for(q=0; q<nv; q++) {
      for (i=0; i<closureSize*2; i+=2) {
        if ((closure[i] >= fStart) && (closure[i] < fEnd)) {
          fclosure = NULL;
          ierr = DMPlexGetTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,
                                            &fclosure); CHKERRQ(ierr);
          for(f=0; f<fclosureSize*2; f+=2) {
            if (fclosure[f] == vmap[c*nv+q]) {
              for(v=0; v<fclosureSize*2; v+=2) {
                for(d=0; d<dim; d++) {
                  if (fclosure[v] == vmap[c*nv+local_dirs[q*dim+d]]) {
                    (*map)[c*nv*dim+q*dim+d] = closure[i];
                    if (TDyADotBMinusC(&(N[closure[i]*dim]),&(X[closure[i]*dim]),
                                       &(X[c*dim]),dim) < 0) {
                      (*map)[c*nv*dim+q*dim+d] *= -1;
                      break;
                    }
                  }
                }
              }
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,
                                                &fclosure); CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
#if defined(PETSC_USE_DEBUG)
  for(c=0; c<dim*nv*(cEnd-cStart); c++) {
    if((*map)[c]==0) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
              "Unable to find map(cell,local_vertex,dir) -> face");
    }
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode SetQuadrature(PetscQuadrature q,PetscInt dim) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscReal *x,*w;
  PetscInt d,nv=1;
  for(d=0; d<dim; d++) nv *= 2;
  ierr = PetscMalloc1(nv*dim,&x); CHKERRQ(ierr);
  ierr = PetscMalloc1(nv,&w); CHKERRQ(ierr);
  switch(nv*dim) {
  case 2: /* line */
    x[0] = -1.0; w[0] = 1.0;
    x[1] =  1.0; w[1] = 1.0;
    break;
  case 8: /* quad */
    x[0] = -1.0; x[1] = -1.0; w[0] = 1.0;
    x[2] =  1.0; x[3] = -1.0; w[1] = 1.0;
    x[4] = -1.0; x[5] =  1.0; w[2] = 1.0;
    x[6] =  1.0; x[7] =  1.0; w[3] = 1.0;
    break;
  case 24: /* hex */
    x[0]  = -1.0; x[1]  = -1.0; x[2]  = -1.0; w[0] = 1.0;
    x[3]  =  1.0; x[4]  = -1.0; x[5]  = -1.0; w[1] = 1.0;
    x[6]  = -1.0; x[7]  =  1.0; x[8]  = -1.0; w[2] = 1.0;
    x[9]  =  1.0; x[10] =  1.0; x[11] = -1.0; w[3] = 1.0;
    x[12] = -1.0; x[13] = -1.0; x[14] =  1.0; w[4] = 1.0;
    x[15] =  1.0; x[16] = -1.0; x[17] =  1.0; w[5] = 1.0;
    x[18] = -1.0; x[19] =  1.0; x[20] =  1.0; w[6] = 1.0;
    x[21] =  1.0; x[22] =  1.0; x[23] =  1.0; w[7] = 1.0;
  }
  ierr = PetscQuadratureSetData(q,dim,1,nv,x,w); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  u <-- 1/J DF u
*/
static void Piola(PetscReal *u,PetscReal *DF,PetscReal J,PetscInt dim){
  TDY_START_FUNCTION_TIMER()
  PetscInt i,j;
  PetscReal v[3];
  for(i=0;i<dim;i++){
    v[i] = 0;
    for(j=0;j<dim;j++){
      v[i] += DF[i*dim+j]*u[j];
    }
  }
  for(i=0;i<dim;i++) u[i] = v[i]/J;
  TDY_STOP_FUNCTION_TIMER()
}


/*
  BDM1 basis functions on [-1,1] with degrees of freedom chosen to
  match Wheeler2009. Indices map <-- local_vertex*dim + dir.

  2---3
  |   |
  0---1

 */
void HdivBasisQuad(const PetscReal *x,PetscReal *B,PetscReal *DF,PetscReal J) {
  TDY_START_FUNCTION_TIMER()
  PetscInt i;
  B[ 0] = (-x[0]*x[1] + x[0] + x[1] - 1)*0.25;
  B[ 1] = (x[1]*x[1] - 1)*0.125;
  B[ 2] = (x[0]*x[0] - 1)*0.125;
  B[ 3] = (-x[0]*x[1] + x[0] + x[1] - 1)*0.25;
  B[ 4] = (-x[0]*x[1] + x[0] - x[1] + 1)*0.25;
  B[ 5] = (x[1]*x[1] - 1)*0.125;
  B[ 6] = (-x[0]*x[0] + 1)*0.125;
  B[ 7] = (x[0]*x[1] - x[0] + x[1] - 1)*0.25;
  B[ 8] = (x[0]*x[1] + x[0] - x[1] - 1)*0.25;
  B[ 9] = (-x[1]*x[1] + 1)*0.125;
  B[10] = (x[0]*x[0] - 1)*0.125;
  B[11] = (-x[0]*x[1] - x[0] + x[1] + 1)*0.25;
  B[12] = (x[0]*x[1] + x[0] + x[1] + 1)*0.25;
  B[13] = (-x[1]*x[1] + 1)*0.125;
  B[14] = (-x[0]*x[0] + 1)*0.125;
  B[15] = (x[0]*x[1] + x[0] + x[1] + 1)*0.25;
  for(i=0;i<8;i++) Piola(&(B[2*i]),DF,J,2);
  TDY_STOP_FUNCTION_TIMER()
}

void HdivBasisHex(const PetscReal *x,PetscReal *B,PetscReal *DF,PetscReal J) {
  TDY_START_FUNCTION_TIMER()
  PetscInt i;
  B[ 0] = (x[0]*x[1]*x[2] - x[0]*x[1] - x[0]*x[2] + x[0] - x[1]*x[2] + x[1] + x[2] - 1)*0.125;
  B[ 1] = (x[1]*x[1] - 1)*0.0625;
  B[ 2] = (-x[1]*x[2]*x[2] + x[1] + x[2]*x[2] - 1)*0.0625;
  B[ 3] = (-x[0]*x[0]*x[2] + x[0]*x[0] + x[2] - 1)*0.0625;
  B[ 4] = (x[0]*x[1]*x[2] - x[0]*x[1] - x[0]*x[2] + x[0] - x[1]*x[2] + x[1] + x[2] - 1)*0.125;
  B[ 5] = (x[2]*x[2] - 1)*0.0625;
  B[ 6] = (x[0]*x[0] - 1)*0.0625;
  B[ 7] = (-x[0]*x[1]*x[1] + x[0] + x[1]*x[1] - 1)*0.0625;
  B[ 8] = (x[0]*x[1]*x[2] - x[0]*x[1] - x[0]*x[2] + x[0] - x[1]*x[2] + x[1] + x[2] - 1)*0.125;
  B[ 9] = (x[0]*x[1]*x[2] - x[0]*x[1] - x[0]*x[2] + x[0] + x[1]*x[2] - x[1] - x[2] + 1)*0.125;
  B[10] = (x[1]*x[1] - 1)*0.0625;
  B[11] = (-x[1]*x[2]*x[2] + x[1] + x[2]*x[2] - 1)*0.0625;
  B[12] = (x[0]*x[0]*x[2] - x[0]*x[0] - x[2] + 1)*0.0625;
  B[13] = (-x[0]*x[1]*x[2] + x[0]*x[1] + x[0]*x[2] - x[0] - x[1]*x[2] + x[1] + x[2] - 1)*0.125;
  B[14] = (x[2]*x[2] - 1)*0.0625;
  B[15] = (-x[0]*x[0] + 1)*0.0625;
  B[16] = (x[0]*x[1]*x[1] - x[0] + x[1]*x[1] - 1)*0.0625;
  B[17] = (-x[0]*x[1]*x[2] + x[0]*x[1] + x[0]*x[2] - x[0] - x[1]*x[2] + x[1] + x[2] - 1)*0.125;
  B[18] = (-x[0]*x[1]*x[2] + x[0]*x[1] - x[0]*x[2] + x[0] + x[1]*x[2] - x[1] + x[2] - 1)*0.125;
  B[19] = (-x[1]*x[1] + 1)*0.0625;
  B[20] = (x[1]*x[2]*x[2] - x[1] + x[2]*x[2] - 1)*0.0625;
  B[21] = (-x[0]*x[0]*x[2] + x[0]*x[0] + x[2] - 1)*0.0625;
  B[22] = (x[0]*x[1]*x[2] - x[0]*x[1] + x[0]*x[2] - x[0] - x[1]*x[2] + x[1] - x[2] + 1)*0.125;
  B[23] = (x[2]*x[2] - 1)*0.0625;
  B[24] = (x[0]*x[0] - 1)*0.0625;
  B[25] = (x[0]*x[1]*x[1] - x[0] - x[1]*x[1] + 1)*0.0625;
  B[26] = (-x[0]*x[1]*x[2] + x[0]*x[1] - x[0]*x[2] + x[0] + x[1]*x[2] - x[1] + x[2] - 1)*0.125;
  B[27] = (-x[0]*x[1]*x[2] + x[0]*x[1] - x[0]*x[2] + x[0] - x[1]*x[2] + x[1] - x[2] + 1)*0.125;
  B[28] = (-x[1]*x[1] + 1)*0.0625;
  B[29] = (x[1]*x[2]*x[2] - x[1] + x[2]*x[2] - 1)*0.0625;
  B[30] = (x[0]*x[0]*x[2] - x[0]*x[0] - x[2] + 1)*0.0625;
  B[31] = (-x[0]*x[1]*x[2] + x[0]*x[1] - x[0]*x[2] + x[0] - x[1]*x[2] + x[1] - x[2] + 1)*0.125;
  B[32] = (x[2]*x[2] - 1)*0.0625;
  B[33] = (-x[0]*x[0] + 1)*0.0625;
  B[34] = (-x[0]*x[1]*x[1] + x[0] - x[1]*x[1] + 1)*0.0625;
  B[35] = (x[0]*x[1]*x[2] - x[0]*x[1] + x[0]*x[2] - x[0] + x[1]*x[2] - x[1] + x[2] - 1)*0.125;
  B[36] = (-x[0]*x[1]*x[2] - x[0]*x[1] + x[0]*x[2] + x[0] + x[1]*x[2] + x[1] - x[2] - 1)*0.125;
  B[37] = (x[1]*x[1] - 1)*0.0625;
  B[38] = (x[1]*x[2]*x[2] - x[1] - x[2]*x[2] + 1)*0.0625;
  B[39] = (x[0]*x[0]*x[2] + x[0]*x[0] - x[2] - 1)*0.0625;
  B[40] = (-x[0]*x[1]*x[2] - x[0]*x[1] + x[0]*x[2] + x[0] + x[1]*x[2] + x[1] - x[2] - 1)*0.125;
  B[41] = (-x[2]*x[2] + 1)*0.0625;
  B[42] = (x[0]*x[0] - 1)*0.0625;
  B[43] = (-x[0]*x[1]*x[1] + x[0] + x[1]*x[1] - 1)*0.0625;
  B[44] = (x[0]*x[1]*x[2] + x[0]*x[1] - x[0]*x[2] - x[0] - x[1]*x[2] - x[1] + x[2] + 1)*0.125;
  B[45] = (-x[0]*x[1]*x[2] - x[0]*x[1] + x[0]*x[2] + x[0] - x[1]*x[2] - x[1] + x[2] + 1)*0.125;
  B[46] = (x[1]*x[1] - 1)*0.0625;
  B[47] = (x[1]*x[2]*x[2] - x[1] - x[2]*x[2] + 1)*0.0625;
  B[48] = (-x[0]*x[0]*x[2] - x[0]*x[0] + x[2] + 1)*0.0625;
  B[49] = (x[0]*x[1]*x[2] + x[0]*x[1] - x[0]*x[2] - x[0] + x[1]*x[2] + x[1] - x[2] - 1)*0.125;
  B[50] = (-x[2]*x[2] + 1)*0.0625;
  B[51] = (-x[0]*x[0] + 1)*0.0625;
  B[52] = (x[0]*x[1]*x[1] - x[0] + x[1]*x[1] - 1)*0.0625;
  B[53] = (-x[0]*x[1]*x[2] - x[0]*x[1] + x[0]*x[2] + x[0] - x[1]*x[2] - x[1] + x[2] + 1)*0.125;
  B[54] = (x[0]*x[1]*x[2] + x[0]*x[1] + x[0]*x[2] + x[0] - x[1]*x[2] - x[1] - x[2] - 1)*0.125;
  B[55] = (-x[1]*x[1] + 1)*0.0625;
  B[56] = (-x[1]*x[2]*x[2] + x[1] - x[2]*x[2] + 1)*0.0625;
  B[57] = (x[0]*x[0]*x[2] + x[0]*x[0] - x[2] - 1)*0.0625;
  B[58] = (-x[0]*x[1]*x[2] - x[0]*x[1] - x[0]*x[2] - x[0] + x[1]*x[2] + x[1] + x[2] + 1)*0.125;
  B[59] = (-x[2]*x[2] + 1)*0.0625;
  B[60] = (x[0]*x[0] - 1)*0.0625;
  B[61] = (x[0]*x[1]*x[1] - x[0] - x[1]*x[1] + 1)*0.0625;
  B[62] = (-x[0]*x[1]*x[2] - x[0]*x[1] - x[0]*x[2] - x[0] + x[1]*x[2] + x[1] + x[2] + 1)*0.125;
  B[63] = (x[0]*x[1]*x[2] + x[0]*x[1] + x[0]*x[2] + x[0] + x[1]*x[2] + x[1] + x[2] + 1)*0.125;
  B[64] = (-x[1]*x[1] + 1)*0.0625;
  B[65] = (-x[1]*x[2]*x[2] + x[1] - x[2]*x[2] + 1)*0.0625;
  B[66] = (-x[0]*x[0]*x[2] - x[0]*x[0] + x[2] + 1)*0.0625;
  B[67] = (x[0]*x[1]*x[2] + x[0]*x[1] + x[0]*x[2] + x[0] + x[1]*x[2] + x[1] + x[2] + 1)*0.125;
  B[68] = (-x[2]*x[2] + 1)*0.0625;
  B[69] = (-x[0]*x[0] + 1)*0.0625;
  B[70] = (-x[0]*x[1]*x[1] + x[0] - x[1]*x[1] + 1)*0.0625;
  B[71] = (x[0]*x[1]*x[2] + x[0]*x[1] + x[0]*x[2] + x[0] + x[1]*x[2] + x[1] + x[2] + 1)*0.125;
  for(i=0;i<24;i++) Piola(&(B[3*i]),DF,J,3);
  TDY_STOP_FUNCTION_TIMER()
}

