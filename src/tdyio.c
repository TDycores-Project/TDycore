#include <private/tdycoreimpl.h>
#include <tdyio.h>
#include "exodusII.h"
#include <petsc/private/dmpleximpl.h>
#include <petscviewerhdf5.h>

PetscErrorCode TDyIOCreate(TDyIO *_io) {
  TDyIO io;
  PetscFunctionBegin;
  io = (TDyIO)malloc(sizeof(struct TDyIO));
  *_io = io;

  io->io_process = PETSC_FALSE;
  io->print_intermediate = PETSC_FALSE;  
  io->num_vars = 1;
  strcpy(io->zonalVarNames[0], "Soln");
  io->format = NullFormat;
  io->num_times = 0;
    
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOSetMode(TDy tdy, TDyIOFormat format){
  PetscFunctionBegin;
  PetscErrorCode ierr;
  
  tdy->io->format = format;
  int num_vars = tdy->io->num_vars;
  DM dm = tdy->dm;
  char *zonalVarNames[1];

  zonalVarNames[0] = tdy->io->zonalVarNames[0];

  if (tdy->io->format == ExodusFormat) {
    strcpy(tdy->io->filename, "out.exo");
    char *ofilename = tdy->io->filename;
    ierr = TdyIOInitializeExodus(ofilename,zonalVarNames,dm,num_vars);CHKERRQ(ierr);
  }
  else if (tdy->io->format == HDF5Format) {
    strcpy(tdy->io->filename, "out.h5");
    char *ofilename = tdy->io->filename;
    ierr = TdyIOInitializeHDF5(ofilename,dm);CHKERRQ(ierr);
  }
    
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteVec(TDy tdy){
  PetscErrorCode ierr;

  int num_vars = tdy->io->num_vars;
  Vec v = tdy->solution;
  DM dm = tdy->dm;
  PetscReal time = tdy->ti->time;
 
  if (tdy->io->format == PetscViewerASCIIFormat) {
    ierr = TDyIOWriteAsciiViewer(v, time);CHKERRQ(ierr);
  }
  else if (tdy->io->format == ExodusFormat) {
    char *ofilename = tdy->io->filename;

    ierr = TdyIOAddExodusTime(ofilename,time,tdy->io);CHKERRQ(ierr);
    ierr = TdyIOWriteExodusVar(ofilename,v,tdy->io);CHKERRQ(ierr);
  }
  else if (tdy->io->format == HDF5Format) {
    char *ofilename = tdy->io->filename;
    ierr = TdyIOWriteHDF5Var(ofilename,v,time);CHKERRQ(ierr);
  }
  else{
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unrecognized IO format, must call TDyIOSetMode");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TdyIOInitializeHDF5(char *ofilename, DM dm){
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

PetscErrorCode TdyIOWriteHDF5Var(char *ofilename, Vec U,PetscReal time){   
  PetscViewer viewer;
  PetscErrorCode ierr;
  char word[32];
  sprintf(word,"%11.5e",time);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,ofilename,FILE_MODE_APPEND,&viewer);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) U,word);CHKERRQ(ierr);
  ierr = VecView(U,viewer);CHKERRQ(ierr);  
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode TdyIOInitializeExodus(char *ofilename, char *zonalVarNames[], DM dm, int num_vars){
  int CPU_word_size, IO_word_size;
  PetscErrorCode ierr;
  int exoid = -1;
  
  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);

  exoid = ex_create(ofilename,EX_CLOBBER, &CPU_word_size, &IO_word_size);
  if (exoid < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to open exodus file %\n", ofilename);

  ierr = DMPlexView_ExodusII_Internal(dm,exoid,1);CHKERRQ(ierr);

  ierr = ex_put_variable_param(exoid, EX_ELEM_BLOCK, num_vars);CHKERRQ(ierr);
  ierr = ex_put_variable_names(exoid,EX_ELEM_BLOCK, num_vars, zonalVarNames);CHKERRQ(ierr);
  ierr = ex_close(exoid);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TdyIOAddExodusTime(char *ofilename, PetscReal time, TDyIO io){
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
  ierr = ex_close(exoid);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
  
PetscErrorCode TdyIOWriteExodusVar(char *ofilename, Vec U, TDyIO io){ 
  int CPU_word_size, IO_word_size;
  PetscErrorCode ierr;
  float version;
  int exoid = -1;
  
  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);

  exoid = ex_open(ofilename, EX_WRITE, &CPU_word_size, &IO_word_size, &version);
  if (exoid < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to open exodus file %\n", ofilename);
  ierr = PetscObjectSetName((PetscObject) U,  "Soln");CHKERRQ(ierr); 
  ierr = VecViewPlex_ExodusII_Zonal_Internal(U, exoid, io->num_times);CHKERRQ(ierr);       
  ierr = ex_close(exoid);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteAsciiViewer(Vec v,PetscReal time) {
  char word[32];
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  sprintf(word,"%11.5e.txt",time);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,word,&viewer);
         CHKERRQ(ierr);
  ierr = VecView(v,viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIODestroy(TDyIO *io) {
  PetscFunctionBegin;
  free(*io);
  io = PETSC_NULL;
  PetscFunctionReturn(0);
}
