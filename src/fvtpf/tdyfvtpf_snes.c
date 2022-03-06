#include <tdytimers.h>
#include <private/tdycoreimpl.h>
#include <private/tdyfvtpfimpl.h>
#include <private/tdymeshimpl.h>
#include <private/tdyutils.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdydiscretization.h>

/* -------------------------------------------------------------------------- */
PetscErrorCode TDyFVTPFSNESFunction(SNES snes,Vec U,Vec R,void *ctx) {

  TDy      tdy = (TDy)ctx;
  TDyFVTPF *fvtpf = tdy->context;
  TDyMesh  *mesh = fvtpf->mesh;
  TDyCell  *cells = &mesh->cells;
  PetscReal *p,*r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TDY_START_FUNCTION_TIMER()

  printf("Stopping in SNES Function\n");
  exit(0);

  TDY_STOP_FUNCTION_TIMER()

  PetscFunctionReturn(0);
}