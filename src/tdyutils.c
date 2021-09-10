#include <private/tdyutils.h>
#include <petscblaslapack.h>
#include <private/tdymemoryimpl.h>
#include <tdytimers.h>

/* ---------------------------------------------------------------- */
PetscErrorCode Increase_Closure_Array(DM dm, PetscInt **closure, PetscInt *maxClosureSize, PetscInt newSize) {

  PetscInt **tmpArray;
  PetscInt oldSize;
  PetscInt pStart, pEnd, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = DMPlexGetChart(dm, &pStart, &pEnd); CHKERRQ(ierr);

  oldSize = *maxClosureSize;

  ierr = TDyAllocate_IntegerArray_2D(&tmpArray, pEnd, 2*newSize); CHKERRQ(ierr);

  for (i=pStart; i<pEnd; i++) {
    for (j=0; j<*maxClosureSize; j++) {
      tmpArray[i][j] = closure[i][j];
    }
  }

  ierr = TDyDeallocate_IntegerArray_2D(closure, pEnd); CHKERRQ(ierr);

  *maxClosureSize = newSize;
  ierr = TDyAllocate_IntegerArray_2D(&closure, pEnd, 2*newSize); CHKERRQ(ierr);

  for (i=pStart; i<pEnd; i++) {
    for (j=0; j<oldSize; j++) {
      closure[i][j] = tmpArray[i][j];
    }
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}


/* ---------------------------------------------------------------- */
PetscErrorCode TDySaveClosures_Elemnts(DM dm, PetscInt *closureSize, PetscInt **closure, PetscInt *maxClosureSize, PetscInt eStart, PetscInt eEnd, PetscBool use_cone){
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt i, e;
  PetscInt pSize,*p;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);

  for(e=eStart; e<eEnd; e++) {
    p = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,e,use_cone,&pSize,&p);CHKERRQ(ierr);
    closureSize[e] = pSize;

    if (pSize > *maxClosureSize) {
      // Increase the column size by 2 x pSize
      ierr = Increase_Closure_Array(dm, closure, maxClosureSize, 2*pSize); CHKERRQ(ierr);
    }

    for (i=0;i<pSize*2;i++) closure[e][i] = p[i];
    ierr = DMPlexRestoreTransitiveClosure(dm,e,use_cone,&pSize,&p);CHKERRQ(ierr);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDySaveClosures_Cells(DM dm, PetscInt *closureSize, PetscInt **closure, PetscInt *maxClosureSize){
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt cStart, cEnd;
  PetscBool use_cone = PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = TDySaveClosures_Elemnts(dm, closureSize, closure, maxClosureSize, cStart, cEnd, use_cone); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDySaveClosures_Faces(DM dm, PetscInt *closureSize, PetscInt **closure, PetscInt *maxClosureSize){
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt fStart, fEnd;
  PetscBool use_cone = PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = DMPlexGetDepthStratum(dm, 2, &fStart, &fEnd); CHKERRQ(ierr);
  ierr = TDySaveClosures_Elemnts(dm, closureSize, closure, maxClosureSize, fStart, fEnd, use_cone); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDySaveClosures_Vertices(DM dm, PetscInt *closureSize, PetscInt **closure, PetscInt *maxClosureSize){
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt vStart, vEnd;
  PetscBool use_cone = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = TDySaveClosures_Elemnts(dm, closureSize, closure, maxClosureSize, vStart, vEnd, use_cone); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDySaveClosures(DM dm, PetscInt *closureSize, PetscInt **closure, PetscInt *maxClosureSize){
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt dim;
  PetscErrorCode ierr;

  ierr = TDySaveClosures_Cells(dm, closureSize, closure, maxClosureSize); CHKERRQ(ierr);
  ierr = TDySaveClosures_Vertices(dm, closureSize, closure, maxClosureSize); CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = TDySaveClosures_Faces(dm, closureSize, closure, maxClosureSize); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscInt TDyGetNumberOfCellVerticesWithClosures(DM dm, PetscInt *closureSize, PetscInt **closure) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt nq,c,q,i,cStart,cEnd,vStart,vEnd;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  nq = -1;
  for(c=cStart; c<cEnd; c++) {
    q = 0;
    for (i=0; i<closureSize[c]*2; i+=2) {
      if ((closure[c][i] >= vStart) && (closure[c][i] < vEnd)) q += 1;
    }
    if(nq == -1) nq = q;
    if(nq !=  q) SETERRQ(comm,PETSC_ERR_SUP,"Mesh cells must be of uniform type");
  }
  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(nq);
}

/* ---------------------------------------------------------------- */
PetscInt TDyMaxNumOfAElmTypeSharingOtherElmType(PetscInt *closureSize, PetscInt **closure, PetscInt aStart, PetscInt aEnd, PetscInt oStart, PetscInt oEnd) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscInt nElem,a,o,result; //cStart,cEnd,vStart,vEnd,result;

  result = 0;
  for(a=aStart; a<aEnd; a++) {
    nElem = 0;
    for (o=0; o<closureSize[a]*2; o+=2) {
      if ((closure[a][o] >= oStart) && (closure[a][o] < oEnd)) nElem += 1;
    }
    result = PetscMax(result, nElem);
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(result);
}

/* ---------------------------------------------------------------- */
PetscInt TDyMaxNumberOfCellsSharingAVertex(DM dm, PetscInt *closureSize, PetscInt **closure) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt cStart,cEnd,vStart,vEnd,result;

  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);

  result = TDyMaxNumOfAElmTypeSharingOtherElmType(closureSize, closure, vStart, vEnd, cStart, cEnd);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(result);
}

/* ---------------------------------------------------------------- */
PetscInt TDyMaxNumberOfEdgesSharingAVertex(DM dm, PetscInt *closureSize, PetscInt **closure) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt eStart,eEnd,vStart,vEnd,result;

  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,1,&eStart,&eEnd); CHKERRQ(ierr);

  result = TDyMaxNumOfAElmTypeSharingOtherElmType(closureSize, closure, vStart, vEnd, eStart, eEnd);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(result);
}

/* ---------------------------------------------------------------- */
PetscInt TDyMaxNumberOfFacesSharingAVertex(DM dm, PetscInt *closureSize, PetscInt **closure) {
  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscInt fStart,fEnd,vStart,vEnd,result;

  ierr = PetscObjectGetComm((PetscObject)dm,&comm); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,2,&fStart,&fEnd); CHKERRQ(ierr);

  result = TDyMaxNumOfAElmTypeSharingOtherElmType(closureSize, closure, vStart, vEnd, fStart, fEnd);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(result);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyComputeLength(PetscReal v1[3], PetscReal v2[3], PetscInt dim,
                                PetscReal *length) {

  PetscFunctionBegin;
  PetscInt d;
  *length = 0.0;
  for (d=0; d<dim; d++) *length += pow(v1[d] - v2[d], 2.0);
  *length = pow(*length, 0.5);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyCrossProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal cross_P[3])
{
  // Function to compute cross product of two vector array.
  PetscFunctionBegin;

  cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
  cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
  cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyDotProduct(PetscReal vect_A[3], PetscReal vect_B[3], PetscReal *dot_P)
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
PetscErrorCode TDyCreateVecJoiningTwoVertices(PetscReal vtx_from[3],
                                           PetscReal vtx_to[3],
                                           PetscReal vec[3]) {
  PetscFunctionBegin;
  PetscInt d=3;

  for (d=0; d<3; d++) vec[d] = vtx_to[d] - vtx_from[d];

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyTriangleArea(PetscReal node_1[3], PetscReal node_2[3],
                            PetscReal node_3[3], PetscReal *area) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  PetscReal a[3], b[3], axb[3], vec_norm;

  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_2, a); CHKERRQ(ierr);
  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_3, b); CHKERRQ(ierr);

  ierr = TDyCrossProduct(a,b,axb); CHKERRQ(ierr);

  ierr = Norm(axb, &vec_norm); CHKERRQ(ierr);

  *area = 0.5*vec_norm;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyQuadrilateralArea(PetscReal node_1[3], PetscReal node_2[3],
                            PetscReal node_3[3], PetscReal node_4[3], PetscReal *area) {
  TDY_START_FUNCTION_TIMER()
  PetscReal node_cen[3];
  PetscInt  d;
  PetscReal tri_area;
  PetscErrorCode ierr;

  for (d=0; d<3; d++) {
    node_cen[d] = (node_1[d] + node_2[d] + node_3[d] + node_4[d])/4.0;
  }

  *area = 0.;

  ierr = TDyTriangleArea(node_cen, node_1, node_2, &tri_area); CHKERRQ(ierr); *area += tri_area;
  ierr = TDyTriangleArea(node_cen, node_2, node_3, &tri_area); CHKERRQ(ierr); *area += tri_area;
  ierr = TDyTriangleArea(node_cen, node_3, node_4, &tri_area); CHKERRQ(ierr); *area += tri_area;
  ierr = TDyTriangleArea(node_cen, node_4, node_1, &tri_area); CHKERRQ(ierr); *area += tri_area;

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyUnitNormalVectorJoiningTwoVertices(PetscReal node_1[3], PetscReal node_2[3],
                                PetscReal normal[3]) {
  PetscErrorCode ierr;

  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_2, normal); CHKERRQ(ierr);

  ierr = ConvertToUnitVector(normal); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],
                                PetscReal node_3[3], PetscReal normal[3]) {
  TDY_START_FUNCTION_TIMER()
  PetscReal a[3], b[3];
  PetscErrorCode ierr;

  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_2, a); CHKERRQ(ierr);
  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_3, b); CHKERRQ(ierr);

  ierr = TDyCrossProduct(a,b,normal); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyUnitNormalToTriangle(PetscReal node_1[3], PetscReal node_2[3],
                                    PetscReal node_3[3], PetscReal normal[3]) {
  TDY_START_FUNCTION_TIMER()
  PetscErrorCode ierr;

  ierr = TDyNormalToTriangle(node_1, node_2, node_3, normal); CHKERRQ(ierr);
  ierr = ConvertToUnitVector(normal); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyNormalToQuadrilateral(PetscReal node_1[3], PetscReal node_2[3],
                                     PetscReal node_3[3], PetscReal node_4[3],
                                     PetscReal normal[3]) {
  TDY_START_FUNCTION_TIMER()
  PetscReal node_cen[3];
  PetscReal normal_12c[3], normal_23c[3], normal_34c[3], normal_41c[3];
  PetscInt  d;
  PetscErrorCode ierr;

  for (d=0; d<3; d++) {
    node_cen[d] = (node_1[d] + node_2[d] + node_3[d] + node_4[d])/4.0;
  }

  ierr = TDyUnitNormalToTriangle(node_cen, node_1, node_2, normal_12c); CHKERRQ(ierr);
  ierr = TDyUnitNormalToTriangle(node_cen, node_2, node_3, normal_23c); CHKERRQ(ierr);
  ierr = TDyUnitNormalToTriangle(node_cen, node_3, node_4, normal_34c); CHKERRQ(ierr);
  ierr = TDyUnitNormalToTriangle(node_cen, node_4, node_1, normal_41c); CHKERRQ(ierr);

  for (d=0; d<3; d++) {
    normal[d] = (normal_12c[d] + normal_23c[d] + normal_34c[d] + normal_41c[d])/4.0;
  }

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);

}

/* ---------------------------------------------------------------- */
PetscErrorCode TDyComputeVolumeOfTetrahedron(PetscReal node_1[3], PetscReal node_2[3],
                                          PetscReal node_3[3], PetscReal node_4[3],
                                          PetscReal *volume) {

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  PetscReal a[3], b[3], c[3], axb[3], dot_prod;
  PetscErrorCode ierr;

  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_2, a); CHKERRQ(ierr);
  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_3, b); CHKERRQ(ierr);
  ierr = TDyCreateVecJoiningTwoVertices(node_1, node_4, c); CHKERRQ(ierr);

  ierr = TDyCrossProduct(a,b,axb); CHKERRQ(ierr);
  ierr = TDyDotProduct(axb,c,&dot_prod); CHKERRQ(ierr);
  *volume = PetscAbsReal(dot_prod)/6.0; CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscInt TDyReturnIndexInList(PetscInt *list, PetscInt nlist, PetscInt value) {

  PetscFunctionBegin;

  PetscInt idx = -1;
  PetscInt i;

  for (i=0; i<nlist; i++){
    if (list[i] == value) {
      idx = i;
      break;
    }
  }

  if (idx == -1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "TDyReturnIndexInList: Did not find the value in the list");

  PetscFunctionReturn(idx);
}

/* -------------------------------------------------------------------------- */
/// Outputs a sequential PETSc Vec as binary
///
/// @param [in] vec A sequential PETSc Vec
/// @param [in] filename Name of the file
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDySavePetscVecSeqAsBinary(Vec vec, const char filename[]) {

  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, filename, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(vec, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Outputs a PETSc Vec as binary
///
/// @param [in] vec A PETSc Vec
/// @param [in] filename Name of the file
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDySavePetscVecAsBinary(Vec vec, const char filename[]) {

  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE,
                               &viewer); CHKERRQ(ierr);
  ierr = VecView(vec, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Reads a PETSc Vec as binary
///
/// @param [in] vec A PETSc Vec
/// @param [in] filename Name of the file
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyReadBinaryPetscVec(Vec vec, const char filename[]) {

  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
  ierr = VecLoad(vec, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode TDySavePetscMatAsBinary(Mat M, const char filename[]) {

  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE,
                               &viewer); CHKERRQ(ierr);
  ierr = MatView(M, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
PetscErrorCode ExtractSubVectors(Vec A, PetscInt stride, Vec *Asub) {

  PetscInt local_size, block_size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  ierr = VecGetLocalSize(A,&local_size); CHKERRQ(ierr);
  ierr = VecGetBlockSize(A,&block_size); CHKERRQ(ierr);

  if (stride>= block_size)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "ExtractSubVectors: stride > block size");

  ierr = VecCreate(PETSC_COMM_WORLD,Asub); CHKERRQ(ierr);
  ierr = VecSetSizes(*Asub,local_size/block_size,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetFromOptions(*Asub); CHKERRQ(ierr);
  ierr = VecStrideGather(A,stride,*Asub,INSERT_VALUES); CHKERRQ(ierr);

  TDY_STOP_FUNCTION_TIMER()
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode ComputeTheta(PetscReal x, PetscReal y, PetscReal *theta) {

  PetscFunctionBegin;

  if (x>0.0) {
    if (y>= 0.0) *theta = atan(y/x);
    else         *theta = atan(y/x) + 2.0*PETSC_PI;
  } else if (x==0.0) {
    if      (y>  0.0) *theta = 0.5*PETSC_PI;
    else if (y ==0.0) *theta = 0.;
    else              *theta = 1.5*PETSC_PI;
  } else {
    *theta = atan(y/x) + PETSC_PI;
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode ComputeDeterminantOf3by3Matrix(PetscReal a[9], PetscReal *det_a) {  

  PetscFunctionBegin;

  *det_a = a[0]*(a[4]*a[8] - a[5]*a[7]) - a[1]*(a[3]*a[8] - a[5]*a[6]) + a[2]*(a[3]*a[7] - a[4]*a[6]);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode ComputeInverseOf3by3Matrix(PetscReal a[9], PetscReal inv_a[9]) {

  PetscFunctionBegin;

  PetscReal det_a;
  PetscErrorCode ierr;

  ierr = ComputeDeterminantOf3by3Matrix(a, &det_a); CHKERRQ(ierr);

  inv_a[0] = +1.0/det_a*(a[4]*a[8]-a[7]*a[5]);
  inv_a[1] = -1.0/det_a*(a[1]*a[8]-a[7]*a[2]);
  inv_a[2] = +1.0/det_a*(a[1]*a[5]-a[4]*a[2]);
  inv_a[3] = -1.0/det_a*(a[3]*a[8]-a[6]*a[5]);
  inv_a[4] = +1.0/det_a*(a[0]*a[8]-a[6]*a[2]);
  inv_a[5] = -1.0/det_a*(a[0]*a[5]-a[3]*a[2]);
  inv_a[6] = +1.0/det_a*(a[3]*a[7]-a[6]*a[4]);
  inv_a[7] = -1.0/det_a*(a[0]*a[7]-a[6]*a[1]);
  inv_a[8] = +1.0/det_a*(a[0]*a[4]-a[3]*a[1]);

  for (PetscInt ii=0; ii<9; ii++) {
    if (fabs(a[ii]) < PETSC_MACHINE_EPSILON) a[ii] = 0.0;
  }

  PetscFunctionReturn(0);
}
