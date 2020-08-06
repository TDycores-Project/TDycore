#include <private/tdycoreimpl.h>

PetscErrorCode TDyRegressionInitialize(TDy tdy) {

  TDy_regression *regression;
  DM dm;
  PetscInt c;
  PetscInt increment, global_offset;
  PetscInt myrank, size;
  PetscErrorCode ierr;
  Vec U;
  PetscInt ncells_local, min_ncells_local;
  Vec temp_vec;
  PetscScalar *vec_ptr;
  PetscInt global_count;
  PetscInt *int_array;
  IS temp_is;
  PetscBool opt;
  VecScatter temp_scatter;

  PetscFunctionBegin;

  dm = tdy->dm;

  regression = (TDy_regression *) malloc(sizeof(TDy_regression));

  regression->num_cells_per_process = 2;

  ierr = PetscObjectOptionsBegin((PetscObject)tdy); CHKERRQ(ierr);

  ierr = PetscOptionsInt ("-tdy_regression_test_num_cells_per_process",
			  "Number of cells per MPI process","",
			  regression->num_cells_per_process,
			  &(regression->num_cells_per_process),NULL);

  ierr = PetscOptionsGetString(NULL,NULL,"-tdy_regression_test_filename",
			       regression->filename,sizeof(regression->filename),
                               &opt); CHKERRQ(ierr);
  if (!opt) {
    SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,
	    "Need to specify a regression filename via -tdy_regression_test_filename");
  }
  
  MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);
  MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&myrank);

  ierr = DMCreateGlobalVector(dm,&U); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U,&ncells_local); CHKERRQ(ierr);

  ierr = MPI_Allreduce(&ncells_local,&min_ncells_local,1,MPIU_INT,MPI_MIN,PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);

  if (min_ncells_local<regression->num_cells_per_process) {
    regression->num_cells_per_process = min_ncells_local;
  }

  if (myrank == 0) {
    ierr = PetscMalloc(ncells_local*regression->num_cells_per_process*sizeof(PetscInt),&(regression->cells_per_process_natural_ids)); CHKERRQ(ierr);
  }
 
  if (tdy->mode == TH) {
    increment = floor(ncells_local/regression->num_cells_per_process/2);
  }
  else {
    increment = floor(ncells_local/regression->num_cells_per_process);
  }
  

  global_offset = 0;
  ierr = MPI_Exscan(&ncells_local,&global_offset,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);


  ierr = VecCreate(PetscObjectComm((PetscObject)dm),&temp_vec); CHKERRQ(ierr);
  if (tdy->mode == TH) {
    ierr = VecSetSizes(temp_vec,2*regression->num_cells_per_process,PETSC_DECIDE); CHKERRQ(ierr);
  }
  else {
    ierr = VecSetSizes(temp_vec,regression->num_cells_per_process,PETSC_DECIDE); CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(temp_vec); CHKERRQ(ierr);
  ierr = VecGetArray(temp_vec,&vec_ptr); CHKERRQ(ierr);


  if (tdy->mode == TH){
    for (c=0; c<regression->num_cells_per_process; c++){
      vec_ptr[2*c] = 2*c*increment + global_offset;
      vec_ptr[2*c+1] = 2*c*increment + global_offset + 1;
    }
  }
  else {
    for (c=0; c<regression->num_cells_per_process; c++){
      vec_ptr[c] = c*increment + global_offset;
    }
  }

  ierr = VecRestoreArray(temp_vec,&vec_ptr); CHKERRQ(ierr);

  ierr = VecGetSize(temp_vec,&global_count); CHKERRQ(ierr);
  if (myrank != 0) global_count = 0;

  ierr = VecCreate(PETSC_COMM_SELF,&(regression->cells_per_process_vec));CHKERRQ(ierr);
  ierr = VecSetSizes(regression->cells_per_process_vec,global_count,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(regression->cells_per_process_vec); CHKERRQ(ierr);

  ierr = PetscMalloc(global_count*sizeof(PetscInt),&(int_array)); CHKERRQ(ierr);

  for (c=0; c<global_count; c++){
    int_array[c] = c;
  }
  
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),global_count,int_array,PETSC_COPY_VALUES,&temp_is); CHKERRQ(ierr);

  ierr = VecScatterCreate(temp_vec,temp_is,regression->cells_per_process_vec,NULL,&temp_scatter); CHKERRQ(ierr);
  ierr = ISDestroy(&temp_is); CHKERRQ(ierr);

  ierr = VecScatterBegin(temp_scatter,temp_vec,regression->cells_per_process_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(temp_scatter,temp_vec,regression->cells_per_process_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&temp_scatter); CHKERRQ(ierr);
  ierr = VecDestroy(&temp_vec); CHKERRQ(ierr);

  if (myrank==0) {
    ierr = VecGetArray(regression->cells_per_process_vec,&vec_ptr); CHKERRQ(ierr);
    for (c=0; c<global_count; c++) {
      regression->cells_per_process_natural_ids[c] = floor(vec_ptr[c]);
      int_array[c] = floor(vec_ptr[c]);
    }
    ierr = VecRestoreArray(regression->cells_per_process_vec,&vec_ptr); CHKERRQ(ierr);
  }
    
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),global_count,int_array,PETSC_COPY_VALUES,&temp_is); CHKERRQ(ierr);

  ierr = VecScatterCreate(U,temp_is,regression->cells_per_process_vec,NULL,&(regression->scatter_cells_per_process_gtos)); CHKERRQ(ierr);
  ierr = ISDestroy(&temp_is); CHKERRQ(ierr);
  
  tdy->regression = regression;

  // Cleanup
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  ierr = PetscFree(int_array); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyRegressionOutput(TDy tdy, Vec U) {

  DM dm;
  TDy_regression *reg;
  PetscInt myrank, size;
  PetscErrorCode ierr;
  PetscInt c;
  Vec U_pres, U_temp;

  PetscFunctionBegin;

  reg = tdy->regression;
  dm = tdy->dm;

  ierr = VecScatterBegin(reg->scatter_cells_per_process_gtos,U,reg->cells_per_process_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(reg->scatter_cells_per_process_gtos,U,reg->cells_per_process_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);


  MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);
  MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&myrank);

  PetscScalar *vec_ptr;
  PetscScalar min_pres_val, max_pres_val, mean_pres_val;
  PetscScalar min_temp_val, max_temp_val, mean_temp_val;
  PetscInt pres_vec_size, temp_vec_size, vec_local_size;


  PetscSection sec;
  PetscInt num_fields;
  ierr = DMGetSection(dm,&sec); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(sec,&num_fields); CHKERRQ(ierr);

  PetscReal *temp_p, *pres_p, *u_p;
  ierr = VecGetLocalSize(U,&vec_local_size); CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm),&U_pres); CHKERRQ(ierr);
  if (tdy->mode == TH) {
    vec_local_size = vec_local_size/2;
  }
  ierr = VecSetSizes(U_pres,vec_local_size,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetFromOptions(U_pres); CHKERRQ(ierr);
  ierr = VecGetArray(U_pres,&pres_p); CHKERRQ(ierr);

  if (tdy->mode == TH) {
    ierr = VecCreate(PetscObjectComm((PetscObject)dm),&U_temp); CHKERRQ(ierr);
    ierr = VecSetSizes(U_temp,vec_local_size,PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetFromOptions(U_temp); CHKERRQ(ierr);
    ierr = VecGetArray(U_temp,&temp_p); CHKERRQ(ierr);
  }


  ierr = VecGetArray(U,&u_p); CHKERRQ(ierr);
  
  if (tdy->mode == TH && num_fields == 2) {
    for (c=0;c<vec_local_size;c++) {
      pres_p[c] = u_p[c*2];
      temp_p[c] = u_p[c*2+1];
    }
  } else {
    for (c=0;c<vec_local_size;c++) {
      pres_p[c] = u_p[c];
    }
  }

  ierr = VecRestoreArray(U,&u_p); CHKERRQ(ierr);
  ierr = VecRestoreArray(U_pres,&pres_p); CHKERRQ(ierr);
 
  if (tdy->mode == TH) { 
    ierr = VecRestoreArray(U_temp,&temp_p); CHKERRQ(ierr);
  }
  
  ierr = VecMax(U_pres,NULL,&max_pres_val); CHKERRQ(ierr);
  ierr = VecMin(U_pres,NULL,&min_pres_val); CHKERRQ(ierr);
  ierr = VecSum(U_pres,&mean_pres_val); CHKERRQ(ierr);
  ierr = VecGetSize(U_pres,&pres_vec_size); CHKERRQ(ierr);
  mean_pres_val = mean_pres_val/pres_vec_size;

  if (tdy->mode == TH) {
    ierr = VecMax(U_temp,NULL,&max_temp_val); CHKERRQ(ierr);
    ierr = VecMin(U_temp,NULL,&min_temp_val); CHKERRQ(ierr);
    ierr = VecSum(U_temp,&mean_temp_val); CHKERRQ(ierr);
    ierr = VecGetSize(U_temp,&temp_vec_size); CHKERRQ(ierr);
    mean_temp_val= mean_temp_val/temp_vec_size;
  }


  if (myrank==0) {
    FILE *fp;
    char filename[267];
    sprintf(filename,"%s.regression",reg->filename);
    if ((fp = fopen(filename,"w")) == NULL) {
      SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_USER,"Unable to write regression file");
    }

    PetscInt count, i;

    ierr = VecGetSize(reg->cells_per_process_vec,&count); CHKERRQ(ierr);
    ierr = VecGetArray(reg->cells_per_process_vec,&vec_ptr); CHKERRQ(ierr);
    fprintf(fp,"-- PRESSURE: Liquid Pressure --\n");
    fprintf(fp,"      Max: %21.13e\n",max_pres_val);
    fprintf(fp,"      Min: %21.13e\n",min_pres_val);
    fprintf(fp,"     Mean: %21.13e\n",mean_pres_val);
   
    if (tdy->mode == TH) { 
      for (i=0; i<count/2; i++) {
        fprintf(fp,"%9d: %21.13e\n",reg->cells_per_process_natural_ids[2*i]/2,vec_ptr[2*i]);
      }
    } else {
      for (i=0; i<count; i++) {
        fprintf(fp,"%9d: %21.13e\n",reg->cells_per_process_natural_ids[i],vec_ptr[i]);
      }
    }
   
    if (tdy->mode == TH) {
      fprintf(fp,"-- GENERIC: Temperature --\n");
      fprintf(fp,"      Max: %21.13e\n",max_temp_val);
      fprintf(fp,"      Min: %21.13e\n",min_temp_val);
      fprintf(fp,"     Mean: %21.13e\n",mean_temp_val);
      for (i=0; i<count/2; i++) {
        fprintf(fp,"%9d: %21.13e\n",reg->cells_per_process_natural_ids[2*i]/2,vec_ptr[2*i+1]);
      }
    }
 
    ierr = VecRestoreArray(reg->cells_per_process_vec,&vec_ptr); CHKERRQ(ierr);

    fclose(fp);
  }

  ierr = VecDestroy(&U_pres); CHKERRQ(ierr);
  if (tdy->mode == TH) ierr = VecDestroy(&U_temp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
