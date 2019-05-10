#include <private/tdycoreimpl.h>

void PressureSaturation_Gardner(PetscReal n,PetscReal m,PetscReal alpha,
                                PetscReal Pc,PetscReal *Se,PetscReal *dSe_dPc) {
  Pc  = PetscMax(Pc,0); /* if Pc < 0 then P > Pref and Se = 1 */
  *Se = PetscExpReal(-alpha*Pc/m);
  if(dSe_dPc) *dSe_dPc = -alpha/m*PetscExpReal(-alpha*Pc/m);
}
