#include <private/tdycoreimpl.h>
#include <private/tdyioimpl.h>
#include <tdyio.h>
#include "exodusII.h"
#include <petsc/private/dmpleximpl.h>

PetscErrorCode TDyIOCreate(TDyIO *_io) {
  TDyIO io;
  PetscFunctionBegin;
  io = (TDyIO)malloc(sizeof(struct _p_TDyIO));
  *_io = io;

  io->io_process = PETSC_FALSE;
  io->print_intermediate = PETSC_FALSE;  
  strcpy(io->exodus_filename, "out.exo");
  io->num_vars = 1;
  strcpy(io->zonalVarNames[0], "Soln");
  io->format = PetscViewerASCIIFormat;
  io->num_times = 0;
    
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOSetIOProcess(TDyIO io, PetscBool flag){
  PetscFunctionBegin;
  io->io_process=flag;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOSetPrintIntermediate(TDyIO io, PetscBool flag){
  PetscFunctionBegin;
  io->print_intermediate=flag;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOSetMode(TDyIO io, TDyIOFormat format){
  PetscFunctionBegin;
  io->format=format;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteVec(TDy tdy){
  PetscErrorCode ierr;
  char *zonalVarNames[1];
 
  char *ofilename = tdy->io->exodus_filename;
  int num_vars = tdy->io->num_vars;
  Vec v = tdy->solution;
  DM dm = tdy->dm = tdy->dm;
  PetscReal time = tdy->ti->time;
  zonalVarNames[0] = tdy->io->zonalVarNames[0];
  
  if (tdy->io->format == PetscViewerASCIIFormat) {
    TDyIOPrintVec(v, time);
  }
  if (tdy->io->format == ExodusFormat){
    if (tdy->io->num_times == 0) {
      TdyIOInitializeExodus(ofilename,zonalVarNames,dm,num_vars);
      ierr = PetscObjectSetName((PetscObject) v,  "Soln");CHKERRQ(ierr);
    }
    TdyIOAddExodusTime(ofilename,time,tdy->io);
    TdyIOWriteExodusVar(ofilename,v,tdy->io);
  }
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
  ierr = VecViewPlex_ExodusII_Zonal_Internal(U, exoid, io->num_times);CHKERRQ(ierr);       
  ierr = ex_close(exoid);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOPrintVec(Vec v,PetscReal time) {
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
