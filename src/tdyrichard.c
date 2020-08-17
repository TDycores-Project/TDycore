#include <tdypermeability.h>
#include <tdyporosity.h>
#include <tdyrichards.h>
#include <private/tdycoreimpl.h>

PetscErrorCode TDyRichardsSNESPostCheck(SNESLineSearch linesearch,
                                        Vec X, Vec Y, Vec W,
                                        PetscBool *changed_Y,
                                        PetscBool *changed_W,void *ctx) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
