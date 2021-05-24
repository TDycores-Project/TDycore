#include <private/tdycoreimpl.h>
#include <private/tdyioimpl.h>
#include <tdyio.h>
#if defined(PETSC_HAVE_EXODUSII)
#include "exodusII.h"
#endif
#include <petsc/private/dmpleximpl.h>
#include <petscviewerhdf5.h>

PetscErrorCode TDyIOCreate(TDyIO *_io) {
  TDyIO io;
  PetscFunctionBegin;
  io = (TDyIO)malloc(sizeof(struct _p_TDyIO));
  *_io = io;

  io->io_process = PETSC_FALSE;
  io->print_intermediate = PETSC_FALSE;  
  io->num_vars = 2;
  strcpy(io->zonalVarNames[0], "LiquidPressure");
  strcpy(io->zonalVarNames[1], "Saturation");
  io->format = NullFormat;
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
    ierr = VecGetSize(tdy->solution, &numCell);CHKERRQ(ierr);
      
    ierr = TDyIOInitializeHDF5(ofilename,dm);CHKERRQ(ierr);
    ierr = TDyIOWriteXMFHeader(numCell,dim,numVert,numCorner);CHKERRQ(ierr);
  }
    
  PetscFunctionReturn(0);
}

PetscErrorCode TDyIOWriteVec(TDy tdy){
  PetscErrorCode ierr;
  PetscBool useNatural;
  Vec s;
  PetscReal *s_vec_ptr;
  PetscInt cStart,cEnd,c,i,n;
  Vec p = tdy->solution;
  DM dm = tdy->dm;
  PetscReal time = tdy->ti->time;
  int num_vars = tdy->io->num_vars;
  char *zonalVarNames[num_vars];

  for (n=0;n<num_vars;++n){
    zonalVarNames[n] =  tdy->io->zonalVarNames[n];
  }
  
  ierr = DMCreateGlobalVector(dm, &s);
  ierr = VecGetArray(s,&s_vec_ptr);
  ierr = DMPlexGetHeightStratum(tdy->dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  for (c=cStart;c<cEnd;c++) {
    i = c-cStart;
    s_vec_ptr[i] = tdy->cc->S[i];
  }
  ierr = VecRestoreArray(s,&s_vec_ptr);CHKERRQ(ierr);

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
      ierr = DMCreateGlobalVector(dm, &p_natural);
      ierr = DMPlexGlobalToNaturalBegin(dm, p, p_natural);CHKERRQ(ierr);
      ierr = DMPlexGlobalToNaturalEnd(dm, p, p_natural);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(dm, &s_natural);
      ierr = DMPlexGlobalToNaturalBegin(dm, s, s_natural);CHKERRQ(ierr);
      ierr = DMPlexGlobalToNaturalEnd(dm, s, s_natural);CHKERRQ(ierr);
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
  ierr = DMGetSection(dm, &sec);
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

  const char *cellMap[24] = {"0"};
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
