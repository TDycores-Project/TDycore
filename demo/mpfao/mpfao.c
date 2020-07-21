#include "tdycore.h"
#include "/home/rosie/software/petsc/arch-linux2-c-opt/externalpackages/git.exodusii/packages/seacas/libraries/exodus/include/exodusII.h"

PetscReal alpha = 1;

void Permeability(double *x,double *K) {
  K[0] = 5; K[1] = 1;
  K[2] = 1; K[3] = 2;
}

PetscErrorCode PermeabilityFunction(TDy tdy, double *x, double *K, void *ctx){
  Permeability(x, K);
  return 0;
}

void Permeability3D(double *x,double *K) {
  K[0] = 1.0; K[1] = 0.0; K[2] = 0.0;
  K[3] = 0.0; K[4] = 1.0; K[5] = 0.0;
  K[6] = 0.0; K[7] = 0.0; K[8] = 1.0;
}

PetscErrorCode PermeabilityFunction3D(TDy tdy, double *x, double *K, void *ctx){
  Permeability3D(x, K);
  return 0;
}

/*--- -dim {2|3} -problem 1 ---------------------------------------------------------------*/

PetscErrorCode PressureConstantFunction(TDy tdy, double *x,double *p, void *ctx) { (*p) = 3.14; return 0; }
PetscErrorCode VelocityConstantFunction(TDy tdy, double *x,double *v, void *ctx) { v[0] = 0; v[1] = 0; v[2] = 0; return 0;}
PetscErrorCode ForcingConstantFunction(TDy tdy, double *x,double *f, void *ctx) { (*f) = 0; return 0;}

/*--- -dim 2 -problem 2 ---------------------------------------------------------------*/
PetscErrorCode PressureQuadraticFunction(TDy tdy, double *x,double *p, void *ctx) { (*p) = 3.14+x[0]*(1-x[0])+x[1]*(1-x[1]); return 0;}
PetscErrorCode VelocityQuadraticFunction(TDy tdy, double *x,double *v, void *ctx) {
  double K[4]; Permeability(x,K);
  v[0] = -K[0]*(1-2*x[0]) - K[1]*(1-2*x[1]);
  v[1] = -K[2]*(1-2*x[0]) - K[3]*(1-2*x[1]);
  return 0;
}
PetscErrorCode ForcingQuadraticFunction(TDy tdy, double *x,double *f, void *ctx) { double K[4]; Permeability(x,K); (*f) = 2*K[0]+2*K[3]; return 0;}

/*--- -dim 2 -problem 3 ---------------------------------------------------------------*/

PetscErrorCode PressureFunction(TDy tdy, double *x, double *f, void *ctx){
  (*f)  = PetscPowReal(1-x[0],4);
  (*f) += PetscPowReal(1-x[1],3)*(1-x[0]);
  (*f) += PetscSinReal(1-x[1])*PetscCosReal(1-x[0]);
  return 0;
}

PetscErrorCode VelocityFunction(TDy tdy, double *x, double *v, void *ctx){
  double vx,vy,K[4];
  Permeability(x,K);
  vx  = -4*PetscPowReal(1-x[0],3);
  vx += -PetscPowReal(1-x[1],3);
  vx += +PetscSinReal(x[1]-1)*PetscSinReal(x[0]-1);
  vy  = -3*PetscPowReal(1-x[1],2)*(1-x[0]);
  vy += -PetscCosReal(x[0]-1)*PetscCosReal(x[1]-1);
  v[0] = -(K[0]*vx+K[1]*vy);
  v[1] = -(K[2]*vx+K[3]*vy);
  return 0;
}

PetscErrorCode ForcingFunction(TDy tdy, double *x, double *f, void *ctx){
  double K[4];
  Permeability(x,K);
  (*f)  = -K[0]*(12*PetscPowReal(1-x[0],
                                 2)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  (*f) += -K[1]*( 3*PetscPowReal(1-x[1],
                                 2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[2]*( 3*PetscPowReal(1-x[1],
                                 2)+PetscSinReal(x[0]-1)*PetscCosReal(x[1]-1));
  (*f) += -K[3]*(-6*(1-x[0])*(x[1]-1)+PetscSinReal(x[1]-1)*PetscCosReal(x[0]-1));
  return 0;
}

/*--- -dim 2 -problem 4 ---------------------------------------------------------------*/

void PermeabilitySine(double *x,double *K) {
  K[0] = 2   ; K[1] = 1.25;
  K[2] = 1.25; K[3] = 3;
}

PetscErrorCode PermeabilitySineFunction(TDy tdy, double *x,double *K, void *ctx) {
  PermeabilitySine(x,K);
  return 0;
}

PetscErrorCode PressureSineFunction(TDy tdy, double *x,double *p, void *ctx) {
  PetscReal s = PetscSinReal(3*PETSC_PI*x[0]);
  PetscReal t = PetscSinReal(3*PETSC_PI*x[1]);
  (*p) = s*s*t*t;
  return 0;
}
PetscErrorCode VelocitySineFunction(TDy tdy, double *x,double *v, void *ctx) {
  double K[4]; PermeabilitySine(x,K);
  double pi = PETSC_PI;
  double s3pX = PetscSinReal(3*pi*x[0]);
  double s3pY = PetscSinReal(3*pi*x[1]);
  double c3pX = PetscCosReal(3*pi*x[0]);
  double c3pY = PetscCosReal(3*pi*x[1]);
  v[0] = -6*K[0]*pi*s3pX*s3pY*s3pY*c3pX - 6*K[1]*pi*s3pX*s3pX*s3pY*c3pY;
  v[1] = -6*K[2]*pi*s3pX*s3pY*s3pY*c3pX - 6*K[3]*pi*s3pX*s3pX*s3pY*c3pY;
  return 0;
}
PetscErrorCode ForcingSineFunction(TDy tdy, double *x,double *f, void *ctx) {
  double K[4]; PermeabilitySine(x,K);
  double pi = PETSC_PI;
  double s3pX = PetscSinReal(3*pi*x[0]);
  double s3pY = PetscSinReal(3*pi*x[1]);
  double c3pX = PetscCosReal(3*pi*x[0]);
  double c3pY = PetscCosReal(3*pi*x[1]);
  (*f) =  18*pi*pi*(K[0]*s3pX*s3pX*s3pY*s3pY - K[0]*s3pY*s3pY*c3pX*c3pX - K[1]*
                    (PetscCosReal(6*pi*(x[0] - x[1])) - PetscCosReal(6*pi*(x[0] + x[1])))/4 - K[2]*
                    (PetscCosReal(6*pi*(x[0] - x[1])) - PetscCosReal(6*pi*(x[0] + x[1])))/4 +
                    K[3]*s3pX*s3pX*s3pY*s3pY - K[3]*s3pX*s3pX*c3pY*c3pY);
  return 0;
}

/*
void Permeability3D(double *x,double *K) {
  K[0] = alpha; K[1] = 1; K[2] = 1;
  K[3] = 1    ; K[4] = 2; K[5] = 1;
  K[6] = 1    ; K[7] = 1; K[8] = 2;
}
*/

PetscErrorCode Pressure3D(TDy tdy, double *x,double *p, void *ctx) {
  (*p)  = (x[0])*(-x[0]+1);
  (*p) += (x[1])*(-x[1]+1);
  (*p) += (x[2])*(-x[2]+1);

  return 0;
}

PetscErrorCode Velocity3D(TDy tdy, double *x,double *v, void *ctx) {
  double K[9]; Permeability3D(x,K);
  v[0] = -K[0]*(-2*x[0]+1) - K[1]*(-2*x[1]+1) - K[2]*(-2*x[2]+1);
  v[1] = -K[3]*(-2*x[0]+1) - K[4]*(-2*x[1]+1) - K[5]*(-2*x[2]+1);
  v[2] = -K[6]*(-2*x[0]+1) - K[7]*(-2*x[1]+1) - K[8]*(-2*x[2]+1);
  return 0;
}

PetscErrorCode Forcing3D(TDy tdy, double *x,double *f, void *ctx) {
  double K[9]; Permeability3D(x,K);
  (*f) = 2*(K[0] + K[4] + K[8]);
  return 0;
}

PetscErrorCode PerturbInteriorVertices(DM dm,PetscReal h) {
  PetscErrorCode ierr;
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value;

  PetscFunctionBegin;

  ierr = DMGetLabelByNum(dm,2,&label);
  CHKERRQ(ierr); // this is the 'marker' label which marks boundary entities

  ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords); CHKERRQ(ierr);

  for(v=vStart; v<vEnd; v++) {
    ierr = PetscSectionGetOffset(coordSection,v,&offset); CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,v,&value); CHKERRQ(ierr);
    if(value==-1) {
      PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*
                    (h*0.471404); // h*sqrt(2)/3
      PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
      coords[offset  ] += r*PetscCosReal(t);
      coords[offset+1] += r*PetscSinReal(t);
    }
  }

  ierr = VecRestoreArray(coordinates,&coords); CHKERRQ(ierr);

  /*
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, "coordinates.bin", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  VecView(coordinates, viewer); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  */

  PetscFunctionReturn(0);
}



int main(int argc, char **argv) {
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 8, dim = 2, problem = 3;
  PetscInt successful_exit_code=0;
  PetscBool perturb = PETSC_FALSE;
  char      mesh_filename[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc,&argv,(char *)0,0); CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-dim","Problem dimension","",dim,&dim,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-N","Number of elements in 1D","",N,&N,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-problem","Problem number","",problem,&problem,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsBool("-perturb","Perturb interior vertices","",perturb,
                          &perturb,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Permeability scaling","",alpha,&alpha,NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-successful_exit_code","Code passed on successful completion","",
                         successful_exit_code,&successful_exit_code,NULL);
  ierr = PetscOptionsString("-mesh_filename", "The mesh file", "", mesh_filename, mesh_filename, PETSC_MAX_PATH_LEN, NULL); CHKERRQ(ierr);

   ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  size_t len;

  ierr = PetscStrlen(mesh_filename, &len); CHKERRQ(ierr);
  if (!len){
    const PetscInt  faces[3] = {N,N,N  };
    const PetscReal lower[3] = {0.0,0.0,0.0};
    const PetscReal upper[3] = {1.0,1.0,1.0};

    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,
                               NULL,PETSC_TRUE,&dm); CHKERRQ(ierr);
   ierr = DMCreateLabel(dm, "celltype");CHKERRQ(ierr);
    if (perturb) {
      ierr = PerturbInteriorVertices(dm,1./N); CHKERRQ(ierr);
    } else {
      ierr = PerturbInteriorVertices(dm,0.); CHKERRQ(ierr);
    }
  } else {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, mesh_filename, PETSC_TRUE, &dm); CHKERRQ(ierr);
    printf("here");
  }

  //



  //

  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  // Setup problem parameters
  TDy  tdy;
  PetscReal gravity[3];

  ierr = TDyCreate(dm,&tdy); CHKERRQ(ierr);

  gravity[0] = 0.0; gravity[1] = 0.0; gravity[2] = 0.0;
  ierr = TDySetGravityVector(tdy,gravity);

  if (dim == 2) {

    ierr = TDySetPermeabilityFunction(tdy,PermeabilityFunction,NULL); CHKERRQ(ierr);

    ierr = TDySetPermeabilityTensor(tdy,Permeability); CHKERRQ(ierr);
    switch(problem) {
    case 1:
      ierr = TDySetForcingFunction(tdy,ForcingConstantFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureConstantFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFluxFunction(tdy,VelocityConstantFunction,NULL); CHKERRQ(ierr);
      break;
    case 2:
      ierr = TDySetForcingFunction(tdy,ForcingQuadraticFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureQuadraticFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFluxFunction(tdy,VelocityQuadraticFunction,NULL); CHKERRQ(ierr);
      break;
    case 3:
      ierr = TDySetForcingFunction(tdy,ForcingFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFluxFunction(tdy,VelocityFunction,NULL); CHKERRQ(ierr);
      break;
    case 4:
      ierr = TDySetPermeabilityFunction(tdy,PermeabilitySineFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetForcingFunction(tdy,ForcingSineFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletValueFunction(tdy,PressureSineFunction,NULL); CHKERRQ(ierr);
      ierr = TDySetDirichletFluxFunction(tdy,VelocitySineFunction,NULL); CHKERRQ(ierr);
      break;
    }
  } else {
    ierr = TDySetPermeabilityFunction(tdy,PermeabilityFunction3D,NULL); CHKERRQ(ierr);
    ierr = TDySetForcingFunction(tdy,Forcing3D,NULL); CHKERRQ(ierr);
    ierr = TDySetDirichletValueFunction(tdy,Pressure3D,NULL); CHKERRQ(ierr);
    ierr = TDySetDirichletFluxFunction(tdy,Velocity3D,NULL); CHKERRQ(ierr);
  }

  ierr = TDySetDiscretizationMethod(tdy,MPFA_O); CHKERRQ(ierr);
  ierr = TDySetFromOptions(tdy); CHKERRQ(ierr);

  // Compute system
  Mat K;
  Vec U,F;
  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F); CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K); CHKERRQ(ierr);
  ierr = TDyComputeSystem(tdy,K,F); CHKERRQ(ierr);

  // Solve system
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U); CHKERRQ(ierr);

  //
  //Actually export mesh

  int CPU_word_size, IO_word_size, exoid;
  PetscReal *mass_p;
  PetscInt c;

  c=0;

   CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);
  
  exoid = ex_create("out.exo",EX_CLOBBER, &CPU_word_size, &IO_word_size);CHKERRQ(ierr);

  ierr = DMPlexView_ExodusII_Internal(dm,exoid,1);CHKERRQ(ierr);
   ierr = ex_close(exoid);CHKERRQ(ierr);
  // ierr = TDyGetLiquidMassValuesLocal(tdy,&c,mass_p);
  // Vec          coordinates;
  //  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  /*
//working
  int CPU_word_size, IO_word_size, exoid;
  int num_dim,num_nodes,num_elem,num_elem_blk,num_elem_blk1,num_node_sets,num_side_sets, block_id, num_nodes_per_elem,num_elem_var;
  float *x, *y, *z;
  int nx,ny,nz,nt;
  int i,j,k,*connect,id,count;
  int ss_id,num_sides_in_ss,*elem_list,*side_list;
  char *coord_names[3], *varname[1];
  float time_value, *elem_var_vals;
  CPU_word_size = sizeof(float);
  IO_word_size  = sizeof(float);
  nx=3;ny=2;nz=2;
  num_dim = 3;
  num_nodes = (nx+1)*(ny+1)*(nz+1);
  num_elem = nx*ny*nz;
  num_elem_blk=1;
  num_elem_blk1=12;
    num_node_sets=0;  //??
   num_side_sets=0;  //?? 65
   num_nodes_per_elem =8;
  num_elem_var = 11;
  
  block_id =10;
  
    x = (float *) calloc (num_nodes, CPU_word_size);
   y = (float *) calloc (num_nodes, CPU_word_size);
   z = (float *) calloc (num_nodes, CPU_word_size);


   id = 0 ;

   for (k=0;k<nz+1; k++) {
     
     for (j=0; j<ny+1; j++) {
       for (i=0; i<nx+1; i++) {
	 
	 x[id] = i;
	 y[id] = j;
	 z[id] = k;

	 	   id += 1;
       }
     }
   }
      
 
   
   coord_names[0] = "Xcoord";
   coord_names[1] = "Ycoord";
   coord_names[2] = "Zcoord";

   //     exoid = ex_create("rosie.exo",EX_CLOBBER, &CPU_word_size, &IO_word_size);CHKERRQ(ierr);
   
   connect = (int *) calloc(8*num_elem, sizeof(int));

   //nxXny(3*3) nxp1Xnyp1 (4*4) nxp1
   for (k=0;k<nz; k++) {
     for (j=0; j<ny; j++) {
       for (i=0; i<nx; i++) {
	
	 id = (k*nx*ny+j*nx+i)*8;
	 //	 printf("%d\n",id);
	 // printf("%d\n",k*4*4+(j+1)*4+i+1);
	 
	 connect[id] = k*(nx+1)*(ny+1)+j*(nx+1) +i +1;
	   connect[id+1] = k*(nx+1)*(ny+1)+j*(nx+1)+i+1+1;
	   connect[id+2] = k*(nx+1)*(ny+1)+(j+1)*(nx+1)+i+1+1;
	   connect[id+3] =k*(nx+1)*(ny+1)+(j+1)*(nx+1)+i+1;
	   connect[id+4] = (k+1)*(nx+1)*(ny+1)+j*(nx+1)+i+1;
	   connect[id+5] =(k+1)*(nx+1)*(ny+1)+j*(nx+1)+i+1+1;
	   connect[id+6]=(k+1)*(nx+1)*(ny+1)+(j+1)*(nx+1)+i+1+1;
	   connect[id+7]=(k+1)*(nx+1)*(ny+1)+(j+1)*(nx+1)+i+1;

	   // 	   printf("i=%i\n",connect[id]);
       }
     }
   }

   varname[0]="pres";
   time_value = 0.0;
   nt = 1;

   	       elem_var_vals = (float *) calloc (num_elem_blk1, sizeof(float));
	       for (j=0; j<num_elem_blk1; j++) {
		 elem_var_vals[j] = 10.0;
	       }

           exoid = ex_create("rosie2.exo",EX_CLOBBER, &CPU_word_size, &IO_word_size);CHKERRQ(ierr);
	   ierr = ex_put_init(exoid, "test", num_dim,num_nodes,num_elem,num_elem_blk,num_node_sets,num_side_sets);CHKERRQ(ierr);
	     ierr = ex_put_coord(exoid,x,y,z);CHKERRQ(ierr);
	     //   ierr = ex_put_coord_names(exoid,coord_names);CHKERRQ(ierr);
    ierr = ex_put_elem_block(exoid,block_id,"HEX",num_elem_blk1,num_nodes_per_elem,0);CHKERRQ(ierr);

      printf("i=%i\n",connect[id]);
     ierr=ex_put_elem_conn(exoid,block_id,connect);CHKERRQ(ierr);
     ierr = ex_put_var_param(exoid,"e",1);CHKERRQ(ierr);
        ierr = ex_put_var_names(exoid,"e",1,varname);CHKERRQ(ierr);
	ierr = ex_put_time(exoid,nt,&time_value);CHKERRQ(ierr);

	 ierr = ex_put_elem_var(exoid,nt,1,block_id,num_elem_blk1,elem_var_vals);CHKERRQ(ierr);	       
		 
      ierr = ex_close(exoid);CHKERRQ(ierr);
  */
   /*
   ss_id=1;
   num_sides_in_ss=3*3;

   elem_list= (int *) calloc (3*3, sizeof(int));
     side_list = (int *) calloc (3*3, sizeof(int));
   for (i=0;i<num_sides_in_ss;i++){
     side_list[i] = 6;
   }

   count = 0;
   k=3-1;
   for(j=0;j<3;j++){
     for(i=0;i<3;i++){
       elem_list[count++]=k*3*3+j*3+i+1;
     }
   }
   
   //   printf("meow");
   // 	  printf("meow = %f\n",connect[100]);
   */

   //  
   // ierr = ex_put_coord(exoid,x,y,z);CHKERRQ(ierr);
   // ierr = ex_put_coord_names(exoid,coord_names);CHKERRQ(ierr);
   //   ierr = ex_put_elem_block(exoid,block_id,"HEX8",num_elem,num_nodes_per_elem,0);CHKERRQ(ierr);
   //  ierr=ex_put_elem_conn(exoid,block_id,connect);CHKERRQ(ierr);
   //   ierr = ex_put_side_set_param (exoid,ss_id,num_sides_in_ss,0);CHKERRQ(ierr);
   //do concatanted
   //   ierr=ex_put_side_set(exoid,ss_id,elem_list,side_list);CHKERRQ(ierr);
   //  free(connect);
  //  free(elem_list);
  //  free(side_list);
  /*
  ss_id=2;
  count =0;
  k=0;
  
  for (j=0; j<3; j++) {
    for (i=0;i<3;i++) {
      elem_list[count++]=k*3*3+j*2+i+1;
    }
  }

  for (i=0;i<num_sides_in_ss;i++) {
    side_list[i] = 5;
  }
  
  ierr=ex_put_side_set_param(exoid,ss_id,num_sides_in_ss,0);
  ierr=ex_put_side_set(exoid,ss_id,elem_list,side_list);
  ierr = ex_close(exoid);CHKERRQ(ierr);
   */
  //

  PetscReal normp, normv;
  ierr = TDyComputeErrorNorms(tdy,U,&normp,&normv);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%e %e\n",normp,normv);CHKERRQ(ierr);

  ierr = TDyOutputRegression(tdy,U);

  // Cleanup

  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = MatDestroy(&K); CHKERRQ(ierr);
  ierr = TDyDestroy(&tdy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return(successful_exit_code);
}
