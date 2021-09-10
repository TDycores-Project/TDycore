#include <private/tdycoreimpl.h>
#include <private/tdyioimpl.h>
#include <private/tdyutils.h>
#include <tdyio.h>
#include <private/tdydiscretization.h>
#if defined(PETSC_HAVE_EXODUSII)
#include "exodusII.h"
#endif
#include <petscsys.h>
#include <petsc/private/dmpleximpl.h>
#include <petscviewerhdf5.h>
#include <private/tdymemoryimpl.h>

PetscErrorCode TDyIOCreate(TDyIO *_io) {
  PetscFunctionBegin;
  TDyIO io;
  PetscErrorCode ierr;

  io = (TDyIO)malloc(sizeof(struct _p_TDyIO));
  *_io = io;

  io->io_process = PETSC_FALSE;
  io->enable_checkpoint = PETSC_FALSE;
  io->num_vars = 2;
  strcpy(io->zonalVarNames[0], "LiquidPressure");
  strcpy(io->zonalVarNames[1], "LiquidSaturation");
  io->format = NullFormat;
  io->num_times = 0;
  io->checkpoint_timestep_interval = 1;
  io->output_timestep_interval = 0;

  io->permeability_filename[0] = '\0';
  io->porosity_filename[0] = '\0';
  io->ic_filename[0] = '\0';

  io->permeability_dataset[0] = '\0';
  io->porosity_dataset[0] = '\0';
  io->ic_dataset[0] = '\0';
  io->anisotropic_permeability = PETSC_FALSE;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sample Options","");
                           CHKERRQ(ierr);
  ierr = PetscOptionsString("-init_permeability_file",
                            "Input Permeability Filename","",
			    io->permeability_filename,io->permeability_filename,
			    PETSC_MAX_PATH_LEN,NULL);
                            CHKERRQ(ierr);
  ierr = PetscOptionsString("-permeability_dataset",
                            "Input Permeability Dataset Name","",
			    io->permeability_dataset,io->permeability_dataset,
			    PETSC_MAX_PATH_LEN,NULL);
                            CHKERRQ(ierr);
  ierr = PetscOptionsString("-init_porosity_file",
                            "Input Porosity Filename","",
			    io->porosity_filename,io->porosity_filename,
			    PETSC_MAX_PATH_LEN,NULL);
                            CHKERRQ(ierr);
  ierr = PetscOptionsString("-porosity_dataset",
                            "Input Porosity Dataset Name","",
			    io->porosity_dataset,io->porosity_dataset,
			    PETSC_MAX_PATH_LEN,NULL);
                            CHKERRQ(ierr);
  ierr = PetscOptionsString("-ic_file",
                            "Input IC Filename","",
			    io->ic_filename,io->ic_filename,
			    PETSC_MAX_PATH_LEN,NULL);
                            CHKERRQ(ierr);
  ierr = PetscOptionsString("-ic_dataset",
                            "Input IC Dataset Name","",
			    io->ic_dataset,io->ic_dataset,
			    PETSC_MAX_PATH_LEN,NULL);
                            CHKERRQ(ierr);
  ierr = PetscOptionsBool("-anisotropic_perm",
                          "Anisotropic Permeability","",
                          io->anisotropic_permeability,&(io->anisotropic_permeability),NULL);
                          CHKERRQ(ierr);
  ierr = PetscOptionsBool("-enable_checkpoint",
                          "Enable checkpoint output","",
                          io->enable_checkpoint,&(io->enable_checkpoint),NULL);
                          CHKERRQ(ierr);
  ierr = PetscOptionsInt("-checkpoint_timestep_interval",
			  "Value of timestep interval for checkpoint output", NULL,
			  io->checkpoint_timestep_interval,
			  &io->checkpoint_timestep_interval, NULL);
                          CHKERRQ(ierr);
  ierr = PetscOptionsInt("-output_timestep_interval",
			  "Value of timestep interval for output", NULL,
			  io->output_timestep_interval,
			  &io->output_timestep_interval, NULL);
                          CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOSetIOProcess(TDyIO io, PetscBool flag){
  PetscFunctionBegin;
  io->io_process=flag;
  PetscFunctionReturn(0);
}

// This context and function assign diagonal tensors to each cell in
// TDyIOReadPermeability below.
typedef struct {
  PetscInt dim;
  PetscReal *Tx, *Ty, *Tz;
} DiagonalTensors;

static PetscErrorCode DiagonalTensorsCreate(PetscInt dim, PetscReal *Tx,
                                            PetscReal *Ty, PetscReal *Tz,
                                            DiagonalTensors **diag_tensors) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscMalloc(sizeof(DiagonalTensors), diag_tensors); CHKERRQ(ierr);
  (*diag_tensors)->dim = dim;
  (*diag_tensors)->Tx = Tx;
  (*diag_tensors)->Ty = Ty;
  (*diag_tensors)->Tz = Tz;
  PetscFunctionReturn(0);
}

static PetscErrorCode AssignDiagonalTensors(void *context, PetscInt n,
                                            PetscReal *x, PetscReal *tensors) {
  PetscFunctionBegin;
  DiagonalTensors *diag_tensors = context;
  PetscInt dim = diag_tensors->dim;
  PetscInt dim2 = dim*dim;
  memset(tensors, 0, sizeof(dim*dim) * n);
  if (dim == 2) {
    for (PetscInt i = 0; i < n; ++i) {
      tensors[dim2*i] = diag_tensors->Tx[i];
      tensors[dim2*i + 3] = diag_tensors->Ty[i];
    }
  } else { // dim == 3
    for (PetscInt i = 0; i < n; ++i) {
      tensors[dim2*i] = diag_tensors->Tx[i];
      tensors[dim2*i + 4] = diag_tensors->Ty[i];
      tensors[dim2*i + 8] = diag_tensors->Tz[i];
    }
  }
  PetscFunctionReturn(0);
}

static void DiagonalTensorsDestroy(void *context) {
  DiagonalTensors *diag_tensors = context;
  if (diag_tensors->Tx) free(diag_tensors->Tx);
  if (diag_tensors->Ty) free(diag_tensors->Ty);
  if (diag_tensors->Tz) free(diag_tensors->Tz);
  PetscFree(diag_tensors);
}

/// Reads in and sets initial permeability for the TDy solver
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyIOReadPermeability(TDy tdy){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  char VariableName[PETSC_MAX_PATH_LEN-1];
  char VariableNameX[PETSC_MAX_PATH_LEN];
  char VariableNameY[PETSC_MAX_PATH_LEN];
  char VariableNameZ[PETSC_MAX_PATH_LEN];
  PetscInt dim;
  size_t len;
  char *filename = tdy->io->permeability_filename;

  ierr = PetscStrlen(tdy->io->permeability_dataset, &len); CHKERRQ(ierr);
  if (!len){
    strcpy(VariableName, "Permeability");
  } else {
    strcpy(VariableName, tdy->io->permeability_dataset);
  }

  ierr = DMGetDimension(tdy->dm, &dim);
  PetscInt cStart,cEnd;
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  if (tdy->io->anisotropic_permeability) {
    // Set up a function/context that assigns a diagonal anisotropic
    // permeability to each cell.
    sprintf(VariableNameX,"%s%s",VariableName,"X");
    sprintf(VariableNameY,"%s%s",VariableName,"Y");
    sprintf(VariableNameZ,"%s%s",VariableName,"Z");

    PetscReal *Kx, *Ky, *Kz;
    ierr = TDyIOReadVariable(tdy,VariableNameX,filename,&Kx);CHKERRQ(ierr);
    ierr = TDyIOReadVariable(tdy,VariableNameY,filename,&Ky);CHKERRQ(ierr);
    ierr = TDyIOReadVariable(tdy,VariableNameZ,filename,&Kz);CHKERRQ(ierr);
    DiagonalTensors *diag_perms;
    ierr = DiagonalTensorsCreate(dim, Kx, Ky, Kz, &diag_perms); CHKERRQ(ierr);
    ierr = MaterialPropSetPermeability(tdy->matprop, diag_perms,
        AssignDiagonalTensors, DiagonalTensorsDestroy);CHKERRQ(ierr);
  } else {
    PetscReal *K;
    ierr = TDyIOReadVariable(tdy,VariableName,filename,&K);CHKERRQ(ierr);

    // Constant diagonal (yet still anisotropic) permeability. Maybe we should
    // change io->anisotropic_permeability to io->heterogeneous_permeability?
    PetscReal perm[3] = {K[0], K[1], K[2]};
    ierr = MaterialPropSetConstantDiagonalPermeability(tdy->matprop, perm);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// This function is used to assign scalar values to each cell below.
static PetscErrorCode AssignScalars(void *context, PetscInt n, PetscReal *x,
                                    PetscReal *scalars) {
  PetscFunctionBegin;
  PetscReal *values = context;
  for (PetscInt i = 0; i < n; ++i) {
    scalars[i] = values[i];
  }
  PetscFunctionReturn(0);
}
static void ScalarsDestroy(void* context) {
  free(context);
}

/* -------------------------------------------------------------------------- */
/// Reads in and sets initial porosity for the TDy solver
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyIOReadPorosity(TDy tdy){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  char VariableName[PETSC_MAX_PATH_LEN];
  size_t len;
  char *filename = tdy->io->porosity_filename;

  ierr = PetscStrlen(tdy->io->porosity_dataset, &len); CHKERRQ(ierr);
  if (!len){
    strcpy(VariableName, "Porosity");
  } else {
    strcpy(VariableName, tdy->io->porosity_dataset);
  }

  PetscInt cStart,cEnd;
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  PetscInt ncell = (cEnd-cStart);

  PetscReal *Porosity;
  ierr = TDyIOReadVariable(tdy,VariableName,filename,&Porosity);
  PetscInt index[ncell];
  for (PetscInt c = 0;c<=ncell;++c){
     index[c] = c;
  }

  ierr = MaterialPropSetPorosity(tdy->matprop, Porosity, AssignScalars,
                                 ScalarsDestroy);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Reads in and sets initial condition for the TDy solver
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyIOReadIC(TDy tdy){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscViewer viewer;
  Vec u;
  char VariableName[PETSC_MAX_PATH_LEN];
  size_t len;

  ierr = PetscStrlen(tdy->io->ic_dataset, &len); CHKERRQ(ierr);
  if (!len){
    strcpy(VariableName, "IC");
  } else {
    strcpy(VariableName, tdy->io->ic_dataset);
  }

  ierr = VecCreate(PETSC_COMM_WORLD,&u);

  ierr = PetscObjectSetName((PetscObject) u, VariableName);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,tdy->io->ic_filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(u,viewer);CHKERRQ(ierr);

  ierr = TDySetInitialCondition(tdy,u);CHKERRQ(ierr);

  ierr = VecDestroy(&u);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Reads in and sets initial permeability for the TDy solver
///
/// @param [inout] tdy A TDy struct
/// @param [in] VariableName A char that is set as the variable name for the
///                          PETSc vector read in from the HDF5 file
/// @param [in] filename A char that is the filename of the HDF5 file
/// @param [inout] variable A pointer to the values read in from HDF5 file
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyIOReadVariable(TDy tdy, char *VariableName, char *filename, PetscReal **variable){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscViewer viewer;
  Vec u;
  Vec u_local;
  PetscReal *ptr;
  int i,n;

  ierr = VecCreate(PETSC_COMM_WORLD,&u);
  ierr = PetscObjectSetName((PetscObject) u, VariableName);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(u,viewer);CHKERRQ(ierr);

  ierr = TDyCreateLocalVector(tdy, &u_local);
  ierr = TDyNaturaltoLocal(tdy,u,&u_local);CHKERRQ(ierr);

  ierr = VecGetArray(u_local,&ptr);CHKERRQ(ierr);
  ierr = VecGetSize(u_local,&n);CHKERRQ(ierr);

  ierr = TDyAllocate_RealArray_1D(variable,n);CHKERRQ(ierr);
  for (i = 0;i<n;++i){
    (*variable)[i] = ptr[i];
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecRestoreArray(u_local,&ptr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/// Outputs a checkpoint file with initial conditions
///
/// @param [inout] tdy A TDy struct
/// @returns 0 on success, or a non-zero error code on failure
PetscErrorCode TDyIOOutputCheckpoint(TDy tdy){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscViewer viewer;
  Vec p = tdy->soln_prev;
  Vec p_natural;
  PetscReal time = tdy->ti->time;
  char filename[PETSC_MAX_PATH_LEN];

  sprintf(filename,"%11.5e_%s.h5",time,"chk");
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);

  ierr = TDyCreateGlobalVector(tdy,&p_natural);
  ierr = TDyGlobalToNatural(tdy,p,p_natural);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) p_natural,"IC");CHKERRQ(ierr);
  ierr = VecView(p_natural,viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer,NULL,"time step", PETSC_INT, (void *) &tdy->ti->istep);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOSetMode(TDy tdy, TDyIOFormat format){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  int n;

  tdy->io->format = format;
  int num_vars = tdy->io->num_vars;
  DM dm = tdy->dm;
  char *zonalVarNames[num_vars];

  PetscInt dim,istart,iend,numCell,numVert,numCorner;

  for (n=0;n<num_vars;++n){
    zonalVarNames[n] =  tdy->io->zonalVarNames[n];
  }

  if (tdy->io->format == ExodusFormat) {
    strcpy(tdy->io->filename, "out.exo");
    char *ofilename = tdy->io->filename;
    ierr = TDyIOInitializeExodus(ofilename,zonalVarNames,dm,num_vars);CHKERRQ(ierr);
  }
  else if (tdy->io->format == HDF5Format) {
    strcpy(tdy->io->filename, "out.h5");
    char *ofilename = tdy->io->filename;
    numCorner = TDyGetNumberOfCellVertices(dm);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,3,&istart,&iend);CHKERRQ(ierr);
    numVert = iend-istart;
    ierr = VecGetSize(tdy->soln_prev, &numCell);CHKERRQ(ierr);

    ierr = TDyIOInitializeHDF5(ofilename,dm);CHKERRQ(ierr);
    ierr = TDyIOWriteXMFHeader(numCell,dim,numVert,numCorner);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteVec(TDy tdy){
  PetscErrorCode ierr;
  PetscBool useNatural;
  PetscReal *s_vec_ptr;
  Vec p = tdy->soln_prev;
  DM dm = tdy->dm;
  PetscReal time = tdy->ti->time;
  int num_vars = tdy->io->num_vars;
  char *zonalVarNames[num_vars];

  for (PetscInt i=0;i<num_vars;++i){
    zonalVarNames[i] =  tdy->io->zonalVarNames[i];
  }

  // Make sure the diagnostics are up to date.
  ierr = TDyUpdateDiagnostics(tdy); CHKERRQ(ierr);

  // Extract the liquid saturation.
  Vec s;
  ierr = TDyCreateDiagnosticVector(tdy, &s); CHKERRQ(ierr);
  ierr = TDyGetLiquidSaturation(tdy, s);
  CHKERRQ(ierr);

  if (tdy->io->format == PetscViewerASCIIFormat) {
    ierr = TDyIOWriteAsciiViewer(p,time,zonalVarNames[0]);CHKERRQ(ierr);
    ierr = TDyIOWriteAsciiViewer(s,time,zonalVarNames[1]);CHKERRQ(ierr);
  }
  else if (tdy->io->format == ExodusFormat) {
    char *ofilename = tdy->io->filename;

    ierr = TDyIOAddExodusTime(ofilename,time,dm,tdy->io);CHKERRQ(ierr);
    ierr = TDyIOWriteExodusVar(ofilename,p,zonalVarNames[0],tdy->io,time);CHKERRQ(ierr);
    ierr = TDyIOWriteExodusVar(ofilename,s,zonalVarNames[1],tdy->io,time);CHKERRQ(ierr);
  }
  else if (tdy->io->format == HDF5Format) {
    char *ofilename = tdy->io->filename;

    ierr = DMGetUseNatural(dm, &useNatural); CHKERRQ(ierr);
    if (useNatural) {
      Vec p_natural;
      Vec s_natural;
      ierr = TDyCreateGlobalVector(tdy,&p_natural);
      ierr = TDyGlobalToNatural(tdy, p, p_natural);CHKERRQ(ierr);

      ierr = TDyCreateGlobalVector(tdy, &s_natural);
      ierr = TDyGlobalToNatural(tdy, s, s_natural);CHKERRQ(ierr);

      ierr = TDyIOWriteHDF5Var(ofilename,dm,p_natural,zonalVarNames[0],time);CHKERRQ(ierr);
      ierr = TDyIOWriteHDF5Var(ofilename,dm,s_natural,zonalVarNames[1],time);CHKERRQ(ierr);
    }
    else {
      ierr = TDyIOWriteHDF5Var(ofilename,dm,p,zonalVarNames[0],time);CHKERRQ(ierr);
      ierr = TDyIOWriteHDF5Var(ofilename,dm,s,zonalVarNames[1],time);CHKERRQ(ierr);
    }
  }
  else{
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unrecognized IO format, must call TDyIOSetMode");
  }

  // Clean up.
  VecDestroy(&s);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOInitializeHDF5(char *ofilename, DM dm){
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscViewerFormat format;
  format = PETSC_VIEWER_HDF5_XDMF;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,ofilename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = DMView(dm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteHDF5Var(char *ofilename,DM dm,Vec U,char *VariableName,PetscReal time){
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscInt numCell;
  char word[32];
  PetscMPIInt rank;
  PetscSection sec;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  sprintf(word,"%11.5e",time);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,ofilename,FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) U,word);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &sec);
  //Change to field name
  ierr = PetscSectionSetFieldName(sec, 0, VariableName); CHKERRQ(ierr);
  ierr = VecView(U,viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  if (rank == 0){
    ierr = VecGetSize(U, &numCell);CHKERRQ(ierr);
    ierr = TDyIOWriteXMFAttribute(VariableName,word,numCell);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOInitializeExodus(char *ofilename, char *zonalVarNames[], DM dm, int num_vars){
#if defined(PETSC_HAVE_EXODUSII)
  PetscErrorCode ierr;
  int exoid = -1;
  PetscViewer viewer;

  ierr = PetscViewerExodusIIOpen(PETSC_COMM_WORLD,ofilename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerExodusIISetOrder(viewer,1);CHKERRQ(ierr);
  ierr = PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD);
  ierr = DMView(dm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD);

  ierr = PetscViewerExodusIIGetId(viewer,&exoid);CHKERRQ(ierr);
  ierr = ex_put_variable_param(exoid, EX_ELEM_BLOCK, num_vars);CHKERRQ(ierr);
  ierr = ex_put_variable_names(exoid,EX_ELEM_BLOCK, num_vars, zonalVarNames);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "PETSc not compiled with Exodus II support.");
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOAddExodusTime(char *ofilename, PetscReal time, DM dm, TDyIO io){
#if defined(PETSC_HAVE_EXODUSII)
  int CPU_word_size, IO_word_size;
  float version;
  PetscErrorCode ierr;
  int exoid = -1;

  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);

  io->num_times = io->num_times + 1;
  exoid = ex_open(ofilename, EX_WRITE, &CPU_word_size, &IO_word_size, &version);
  if (exoid < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to open exodus file %\n", ofilename);
  ierr = ex_put_time(exoid,io->num_times,&time);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm,io->num_times-1,time);CHKERRQ(ierr);
  ierr = ex_close(exoid);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteExodusVar(char *ofilename, Vec U, char *VariableName, TDyIO io, PetscReal time){
#if defined(PETSC_HAVE_EXODUSII)
  PetscErrorCode ierr;
  PetscViewer       viewer;

  ierr = PetscViewerExodusIIOpen(PETSC_COMM_WORLD,ofilename,FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) U,  VariableName);CHKERRQ(ierr);
  ierr = VecView(U, viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteAsciiViewer(Vec U,PetscReal time,char *VariableName) {
  char word[32];
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  sprintf(word,"%11.5e_%s.txt",time,VariableName);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer);
         CHKERRQ(ierr);
  ierr = VecView(U,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteXMFHeader(PetscInt numCell,PetscInt dim,PetscInt numVert,PetscInt numCorner){

  FILE *fid;

  const char *cellMap[25] = {"0"};
  cellMap[1] = "Polyvertex";
  cellMap[2] = "Polyline";
  cellMap[6] = "Triangle";
  cellMap[8] = "Quadrilateral";
  cellMap[12] = "Tetrahedron";
  cellMap[18] = "Wedge";
  cellMap[24] = "Hexahedron";

  //  xmf_filename = "out.xmf";
  fid = fopen("out.xmf","w");
  fprintf(fid,"<?xml version=\"1.0\" ?>");
  fprintf(fid,"\n<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" [");
  fprintf(fid,"\n<!ENTITY HeavyData \"out.h5\">");
  fprintf(fid,"\n]>");
  fprintf(fid, "\n\n<Xdmf>\n  <Domain Name=\"domain\">");

  //cells
  fprintf(fid,"\n    <DataItem Name=\"cells\"");
  fprintf(fid,"\n              ItemType=\"Uniform\"");
  fprintf(fid,"\n              Format=\"HDF\"");
  fprintf(fid,"\n              NumberType=\"Float\" Precision=\"8\"");
  fprintf(fid,"\n              Dimensions=\"%i %i\">",numCell,numCorner);
  fprintf(fid,"\n      &HeavyData;:/viz/topology/cells");
  fprintf(fid,"\n    </DataItem>");

  //write vertices
  fprintf(fid,"\n    <DataItem Name=\"vertices\"");
  fprintf(fid,"\n              Format=\"HDF\"");

  fprintf(fid,"\n              Dimensions=\"%i %i\">",numVert,dim);
  fprintf(fid,"\n      &HeavyData;:/geometry/vertices");
  fprintf(fid,"\n    </DataItem>");

  //Topology and Geometry
  fprintf(fid,"\n      <Grid Name=\"domain\" GridType=\"Uniform\">");
  fprintf(fid,"\n        <Topology");
  fprintf(fid,"\n           TopologyType=\"%s\"",cellMap[dim*numCorner]);
  fprintf(fid,"\n           NumberOfElements=\"%i\">",numCell);
  fprintf(fid,"\n          <DataItem Reference=\"XML\">");
  fprintf(fid,"\n            /Xdmf/Domain/DataItem[@Name=\"cells\"]");
  fprintf(fid,"\n          </DataItem>");
  fprintf(fid,"\n        </Topology>");

  if (dim > 2) {
    fprintf(fid,"\n        <Geometry GeometryType=\"XYZ\">");
    }
  else {
    fprintf(fid,"\n        <Geometry GeometryType=\"XY\">");
  }
  fprintf(fid,"\n          <DataItem Reference=\"XML\">");
  fprintf(fid,"\n            /Xdmf/Domain/DataItem[@Name=\"vertices\"]");
  fprintf(fid,"\n          </DataItem>");
  fprintf(fid,"\n        </Geometry>");

  fclose(fid);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteXMFAttribute(char* name,char* time,PetscInt numCell){

  FILE *fid;

  //  xmf_filename = "out.xmf";
  fid = fopen("out.xmf","a");

  fprintf(fid,"\n        <Attribute");
  fprintf(fid,"\n           Name=\"%s_%s\"",time,name);
  fprintf(fid,"\n           Type=\"Scalar\"");
  fprintf(fid,"\n           Center=\"Cell\">");

  fprintf(fid,"\n          <DataItem ItemType=\"HyperSlab\"");
  fprintf(fid,"\n                    Dimensions=\"1 %i 1\"",numCell);
  fprintf(fid,"\n                    Type=\"HyperSlab\">");
  fprintf(fid,"\n            <DataItem");
  fprintf(fid,"\n               Dimensions=\"3 3\"");
  fprintf(fid,"\n               Format=\"XML\">");
  fprintf(fid,"\n              0 0 0");//dimension
  fprintf(fid,"\n              1 1 1");

  fprintf(fid,"\n              1 %i 1",numCell);
  fprintf(fid,"\n            </DataItem>");
  fprintf(fid,"\n            <DataItem");
  fprintf(fid,"\n               DataType=\"Float\" Precision=\"8\"");
  fprintf(fid,"\n               Dimensions=\"1 %i 1\"",numCell);
  fprintf(fid,"\n               Format=\"HDF\">");
  fprintf(fid,"\n              &HeavyData;:/cell_fields/%s_%s",time,name);
  fprintf(fid,"\n            </DataItem>");
  fprintf(fid,"\n          </DataItem>");
  fprintf(fid,"\n        </Attribute>");

  fclose(fid);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteXMFFooter(){
  FILE *fid;

  //  xmf_filename = "out.xmf";
  fid = fopen("out.xmf","a");

  fprintf(fid,"\n      </Grid>");
  fprintf(fid,"\n  </Domain>");
  fprintf(fid,"\n</Xdmf>\n");
  fclose(fid);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIODestroy(TDyIO *io) {
  PetscFunctionBegin;
  TDyIOWriteXMFFooter();
  free(*io);
  io = PETSC_NULL;
  PetscFunctionReturn(0);
}
