#include <private/tdyutils.h>
#include <petscblaslapack.h>

/* ---------------------------------------------------------------- */
PetscInt GetNumberOfCellVertices(DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,c,q,i,cStart,cEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nq = -1;
  for(c=cStart; c<cEnd; c++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    q = 0;
    for (i=0; i<closureSize*2; i+=2) {
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh cells must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}

/* ---------------------------------------------------------------- */
PetscInt GetNumberOfFaceVertices(DM dm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,f,q,i,fStart,fEnd,vStart,vEnd,closureSize,*closure;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd); CHKERRQ(ierr);
  nq = -1;
  for(f=fStart; f<fEnd; f++) {
    closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
    q = 0;
    for (i=0; i<closureSize*2; i+=2) {
      if ((closure[i] >= vStart) && (closure[i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh faces must be of uniform type");
    ierr = DMPlexRestoreTransitiveClosure(dm,f,PETSC_TRUE,&closureSize,&closure);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(nq);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeLength(PetscReal v1[3], PetscReal v2[3], PetscInt dim,
                             PetscReal *length) {

  PetscFunctionBegin;
  PetscInt d;
  *length = 0.0;
  for (d=0; d<dim; d++) *length += pow(v1[d] - v2[d], 2.0);
  *length = pow(*length, 0.5);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode CrossProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal cross_P[3])
{
  // Function to compute cross product of two vector array.
  PetscFunctionBegin;

  cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
  cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
  cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode DotProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal *dot_P)
{
  // Function to compute cross product of two vector array.
  PetscFunctionBegin;
  PetscInt d;

  *dot_P = 0.0;

  for (d=0; d<3; d++) *dot_P += vect_A[d]*vect_B[d];

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode Norm(PetscReal vect_A[3], PetscReal *vec_norm)
{
  // Function to compute cross product of two vector array.
  PetscFunctionBegin;
  PetscInt d;

  PetscReal norm_P = 0.0;

  for (d=0; d<3; d++) norm_P += vect_A[d]*vect_A[d];

  *vec_norm = PetscSqrtReal(norm_P);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode ConvertToUnitVector(PetscReal vec[3]) {

  PetscFunctionBegin;

  PetscInt d;
  PetscReal vec_norm;
  PetscErrorCode ierr;

  ierr = Norm(vec, &vec_norm); CHKERRQ(ierr);

  for (d=0; d<3; d++) vec[d] = vec[d]/vec_norm;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode CreateVecJoiningTwoVertices(PetscReal vtx_from[3],
                                           PetscReal vtx_to[3],
                                           PetscReal vec[3]) {
  PetscFunctionBegin;
  PetscInt d=3;

  for (d=0; d<3; d++) vec[d] = vtx_to[d] - vtx_from[d];

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TriangleArea(PetscReal node_1[3], PetscReal node_2[3],
                            PetscReal node_3[3], PetscReal *area) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscReal a[3], b[3], axb[3], vec_norm;

  ierr = CreateVecJoiningTwoVertices(node_1, node_2, a); CHKERRQ(ierr);
  ierr = CreateVecJoiningTwoVertices(node_1, node_3, b); CHKERRQ(ierr);

  ierr = CrossProduct(a,b,axb); CHKERRQ(ierr);

  ierr = Norm(axb, &vec_norm); CHKERRQ(ierr);
  
  *area = 0.5*vec_norm;

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode QuadrilateralArea(PetscReal node_1[3], PetscReal node_2[3],
                            PetscReal node_3[3], PetscReal node_4[3], PetscReal *area) {
  PetscReal node_cen[3];
  PetscInt  d;
  PetscReal tri_area;
  PetscErrorCode ierr;

  for (d=0; d<3; d++) {
    node_cen[d] = (node_1[d] + node_2[d] + node_3[d] + node_4[d])/4.0;
  }

  *area = 0.;
  
  ierr = TriangleArea(node_cen, node_1, node_2, &tri_area); CHKERRQ(ierr); *area += tri_area;
  ierr = TriangleArea(node_cen, node_2, node_3, &tri_area); CHKERRQ(ierr); *area += tri_area;
  ierr = TriangleArea(node_cen, node_3, node_4, &tri_area); CHKERRQ(ierr); *area += tri_area;
  ierr = TriangleArea(node_cen, node_4, node_1, &tri_area); CHKERRQ(ierr); *area += tri_area;

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode NormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],
                                PetscReal node_3[3], PetscReal normal[3]) {
  PetscReal a[3], b[3];
  PetscErrorCode ierr;

  ierr = CreateVecJoiningTwoVertices(node_1, node_2, a); CHKERRQ(ierr);
  ierr = CreateVecJoiningTwoVertices(node_1, node_3, b); CHKERRQ(ierr);

  ierr = CrossProduct(a,b,normal); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode UnitNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],
                                    PetscReal node_3[3], PetscReal normal[3]) {
  PetscErrorCode ierr;

  ierr = NormalToTriangle(node_1, node_2, node_3, normal); CHKERRQ(ierr);
  ierr = ConvertToUnitVector(normal); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode NormalToQuadrilateral(PetscReal node_1[3], PetscReal node_2[3],
                                     PetscReal node_3[3], PetscReal node_4[3],
                                     PetscReal normal[3]) {
  PetscReal node_cen[3];
  PetscReal normal_12c[3], normal_23c[3], normal_34c[3], normal_41c[3];
  PetscInt  d;
  PetscErrorCode ierr;

  for (d=0; d<3; d++) {
    node_cen[d] = (node_1[d] + node_2[d] + node_3[d] + node_4[d])/4.0;
  }

  ierr = UnitNormalToTriangle(node_cen, node_1, node_2, normal_12c); CHKERRQ(ierr);
  ierr = UnitNormalToTriangle(node_cen, node_2, node_3, normal_23c); CHKERRQ(ierr);
  ierr = UnitNormalToTriangle(node_cen, node_3, node_4, normal_34c); CHKERRQ(ierr);
  ierr = UnitNormalToTriangle(node_cen, node_4, node_1, normal_41c); CHKERRQ(ierr);

  for (d=0; d<3; d++) {
    normal[d] = (normal_12c[d] + normal_23c[d] + normal_34c[d] + normal_41c[d])/4.0;
  }

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode ComputeVolumeOfTetrahedron(PetscReal node_1[3], PetscReal node_2[3],
                                          PetscReal node_3[3], PetscReal node_4[3],
                                          PetscReal *volume) {

  PetscFunctionBegin;

  PetscReal a[3], b[3], c[3], axb[3], dot_prod;
  PetscErrorCode ierr;

  ierr = CreateVecJoiningTwoVertices(node_1, node_2, a); CHKERRQ(ierr);
  ierr = CreateVecJoiningTwoVertices(node_1, node_3, b); CHKERRQ(ierr);
  ierr = CreateVecJoiningTwoVertices(node_1, node_4, c); CHKERRQ(ierr);

  ierr = CrossProduct(a,b,axb); CHKERRQ(ierr);
  ierr = DotProduct(axb,c,&dot_prod); CHKERRQ(ierr);
  *volume = PetscAbsReal(dot_prod)/6.0; CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
