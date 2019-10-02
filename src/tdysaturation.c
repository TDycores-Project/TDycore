#include <private/tdycoreimpl.h>

void PressureSaturation_Gardner(PetscReal n,PetscReal m,PetscReal alpha,
                                PetscReal Pc,PetscReal *Se,PetscReal *dSe_dPc) {
  if(Pc < 0) { /* if Pc < 0 then P > Pref and Se = 1 */
    *Se = 1;
    if(dSe_dPc) *dSe_dPc = 0;
  }else{
    *Se = PetscExpReal(-alpha*Pc/m);
    if(dSe_dPc) *dSe_dPc = -alpha/m*PetscExpReal(-alpha*Pc/m);
  }
}

void PressureSaturation_VanGenuchten(PetscReal n,PetscReal m,PetscReal alpha,
				     PetscReal Pc,PetscReal *Se,PetscReal *dSe_dPc) {
  PetscReal pc_alpha,pc_alpha_n,one_plus_pc_alpha_n;
  if(Pc <= 0) { 
    *Se = 1;
    if(dSe_dPc) *dSe_dPc = 0;
  }else{
    n = 1/(1-m);
    pc_alpha = Pc*alpha;
    pc_alpha_n = PetscPowReal(pc_alpha,n);
    one_plus_pc_alpha_n = 1+pc_alpha_n;
    *Se = PetscPowReal(one_plus_pc_alpha_n,-m);
    if(dSe_dPc){
      *dSe_dPc = -m*n*alpha*pc_alpha_n/(pc_alpha*PetscPowReal(one_plus_pc_alpha_n,m+1));
    }
  }
}
