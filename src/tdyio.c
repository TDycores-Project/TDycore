#include <tdyio.h>
#include "exodusII.h"

int nt = 0;

PetscErrorCode TDyIOCreate(TDyIO *_io) {
  TDyIO io;
  PetscFunctionBegin;
  io = (TDyIO)malloc(sizeof(struct TDyIO));
  *_io = io;

  io->io_process = PETSC_FALSE;
  io->print_intermediate = PETSC_FALSE;  
  io->exodus_filename = "out.exo";
  io->exodus_initialized = PETSC_FALSE;
  io->num_vars = 1;
  io->zonalVarNames[0] = "Soln";
  io->format=PetscViewer_Format;
    
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOSetMode(TDyIO io, TDyIOFormat format){
  //PetscValidPointer(io,1);
  PetscFunctionBegin;
  io->format=format;
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteVec(TDyIO io,Vec v,const char *prefix,DM dm,int step,PetscReal time){
  PetscErrorCode ierr;
  //etscValidPointer(io,1);
  //etscFunctionBegin;

  //  TDyIO io;
  // *_io = io;
  char *ofilename = io->exodus_filename;
  char *zonalVarNames = io->zonalVarNames;
  int num_vars = io->num_vars;
  
  if (io->format == PetscViewer_Format) {
    TDyIOPrintVec(v, prefix, step);
  }
  if (io->format == Exodus_Format){
    if (io->exodus_initialized == PETSC_FALSE) {
      TdyIOInitializeExodus(ofilename,zonalVarNames,dm,num_vars);
      io->exodus_initialized = PETSC_TRUE;
      ierr = PetscObjectSetName((PetscObject) v,  "Soln");CHKERRQ(ierr);
    }
        TdyIOAddExodusTime(ofilename, time);
       TdyIOWriteExodusVar(ofilename,v);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TdyIOInitializeExodus(char *ofilename, char *zonalVarNames, DM dm, int num_vars){
  
  int CPU_word_size, IO_word_size, exoid;
  PetscErrorCode ierr;

  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);

  exoid = ex_create(ofilename,EX_CLOBBER, &CPU_word_size, &IO_word_size);

  ierr = DMPlexView_ExodusII_Internal(dm,exoid,1);CHKERRQ(ierr);

  ierr = ex_put_variable_param(exoid, EX_ELEM_BLOCK, num_vars);CHKERRQ(ierr);
  ierr = ex_put_variable_names(exoid,EX_ELEM_BLOCK, num_vars, zonalVarNames);CHKERRQ(ierr);

  ierr = ex_close(exoid);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TdyIOAddExodusTime(char *ofilename, PetscReal time){

  int CPU_word_size, IO_word_size, exoid;
  float version;
  PetscErrorCode ierr;

  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);
  
  nt = nt + 1;  
  exoid = ex_open(ofilename, EX_WRITE, &CPU_word_size, &IO_word_size, &version);
  ierr = ex_put_time(exoid,nt,&time);CHKERRQ(ierr);
  ierr = ex_close(exoid);CHKERRQ(ierr);

}
  
PetscErrorCode TdyIOWriteExodusVar(char *ofilename, Vec U){
  
  int CPU_word_size, IO_word_size, exoid;
  PetscErrorCode ierr;
  float version;


  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);

  exoid = ex_open(ofilename, EX_WRITE, &CPU_word_size, &IO_word_size, &version);
  

  ierr = VecViewPlex_ExodusII_Zonal_Internal(U, exoid, nt);CHKERRQ(ierr);
        
  ierr = ex_close(exoid);CHKERRQ(ierr);



  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOPrintVec(Vec v,const char *prefix, int print_count) {
  char word[32];
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (print_count >= 0)
    sprintf(word,"%s_%d.txt",prefix,print_count);
  else
    sprintf(word,"%s.txt",prefix);
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
