#include <private/tdycoreimpl.h>
#include <private/tdycharacteristiccurvesimpl.h>
#include <private/tdymemoryimpl.h>
#include <petscblaslapack.h>
#include <petsc/private/khash/khash.h>
#include <private/tdyutils.h>

/// This type defines a mapping from point indices to (integer) model indices.
/// We use it to select the saturation or relative permeability model for
/// specific points.
KHASH_MAP_INIT_INT(CharacteristicCurvesPointMap, int)

/// Creates a CharacteristicCurves instance with saturation and relative
/// permeability models for use with specific discretizations.
/// @param [out] cc the initialized instance
PetscErrorCode CharacteristicCurvesCreate(CharacteristicCurves **cc) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = PetscMalloc(sizeof(CharacteristicCurves), cc); CHKERRQ(ierr);
  ierr = SaturationCreate(&((*cc)->saturation)); CHKERRQ(ierr);
  ierr = RelativePermeabilityCreate(&((*cc)->rel_perm)); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Frees the resources associated with the given CharacteristicCurves instance.
PetscErrorCode CharacteristicCurvesDestroy(CharacteristicCurves *cc) {

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = RelativePermeabilityDestroy(cc->rel_perm); CHKERRQ(ierr);
  ierr = SaturationDestroy(cc->saturation); CHKERRQ(ierr);
  ierr = PetscFree(cc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/// Creates a new instance of a saturation model.
PetscErrorCode SaturationCreate(Saturation **sat) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = PetscCalloc(sizeof(Saturation), sat); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Frees all resources associated with a given Saturation instance.
PetscErrorCode SaturationDestroy(Saturation *sat) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  int num_types = (int)(sizeof(sat->points)/sizeof(sat->points[0]));
  for (int type = 0; type < num_types; ++type) {
    SaturationSetType(sat, type, 0, NULL, NULL);
  }
  ierr = PetscFree(sat); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the type of the saturation model used for a give set of points, and its
/// parameters. Point indices and parameters are copied into place.
/// @param [in] sat the Saturation inѕtance
/// @param [in] type the type of the saturation model
/// @param [in] num_points the number of points on which the given saturation
///                        model type operates
/// @param [in] points an array of length num_points containing point indices
/// @param [in] parameters an array of length num_params*num_points, with
///                        parameters[num_params*i] containing the first
///                        parameter for the ith point. The number of parameters
///                        depends on the model used.
PetscErrorCode SaturationSetType(Saturation *sat, SaturationType type,
                                 PetscInt num_points, PetscInt* points,
                                 PetscReal* parameters) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  int num_params = 0;
  if (type == SAT_FUNC_GARDNER) {
    num_params = 3;
  } else if (type == SAT_FUNC_VAN_GENUCHTEN) {
    num_params = 2;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid saturation model type!");
  }

  // If we're changing the number of points for this type, free storage.
  if (sat->num_points[type] != num_points) {
    if (sat->points[type]) {
      ierr = PetscFree(sat->points[type]); CHKERRQ(ierr);
      ierr = PetscFree(sat->parameters[type]); CHKERRQ(ierr);
    }
  }

  // Delete the point maps, which are invalidated by this operation.
  int num_types = (int)(sizeof(sat->points)/sizeof(sat->points[0]));
  for (int type = 0; type < num_types; ++type) {
    if (sat->point_maps[type]) {
      khash_t(CharacteristicCurvesPointMap) *point_map = sat->point_maps[type];
      kh_destroy(CharacteristicCurvesPointMap, point_map);
      sat->point_maps[type] = NULL;
    }
  }

  sat->num_points[type] = num_points;
  ierr = PetscMalloc(num_points*sizeof(PetscInt),
                     &(sat->points[type])); CHKERRQ(ierr);
  memcpy(sat->points[type], points, num_points*sizeof(PetscInt));
  ierr = PetscMalloc(num_params*num_points*sizeof(PetscReal),
                     &(sat->parameters[type])); CHKERRQ(ierr);
  memcpy(sat->parameters[type], parameters, num_params*num_points*sizeof(PetscReal));
  PetscFunctionReturn(0);
}

/// Computes the saturation on all points, according to the type set for each
/// point.
/// @param [in] sat the Saturation inѕtance
/// @param [in] Sr the residual saturation values on the points
/// @param [in] Pc the capillary pressure values on the points
/// @param [out] S the computed saturation values on the points
/// @param [out] dSdP the computed values of the derivative of saturation w.r.t.
///                   pressure on the points
/// @param [out] d2SdP2 the computed values of the second derivative of
///                     saturation w.r.t. pressure on the points
PetscErrorCode SaturationCompute(Saturation *sat,
                                 PetscReal *Sr, PetscReal *Pc,
                                 PetscReal *S, PetscReal *dSdP,
                                 PetscReal *d2SdP2) {
  PetscFunctionBegin;
  int num_types = (int)(sizeof(sat->points)/sizeof(sat->points[0]));
  for (int type = 0; type < num_types; ++type) {
    if (type == SAT_FUNC_GARDNER) {
      PetscInt num_points = sat->num_points[type];
      for (PetscInt i = 0; i < num_points; ++i) {
        PetscInt j = sat->points[type][i];
        PetscReal n = sat->parameters[type][3*i];
        PetscReal m = sat->parameters[type][3*i+1];
        PetscReal alpha = sat->parameters[type][3*i+2];
        PressureSaturation_Gardner(n, m, alpha, Sr[j], Pc[j], &(S[j]), &(dSdP[j]),
                                   &(d2SdP2[j]));
      }
    } else if (type == SAT_FUNC_VAN_GENUCHTEN) {
      PetscInt num_points = sat->num_points[type];
      for (PetscInt i = 0; i < num_points; ++i) {
        PetscInt j = sat->points[type][i];
        PetscReal m = sat->parameters[type][2*i];
        PetscReal alpha = sat->parameters[type][2*i+1];
        PressureSaturation_VanGenuchten(m, alpha, Sr[j], Pc[j], &(S[j]), &(dSdP[j]),
                                        &(d2SdP2[j]));
      }
    } else {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER,
              "Invalid saturation model type!");
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SaturationBuildPointMaps(Saturation *sat) {
  PetscFunctionBegin;
  PetscInt num_types = (PetscInt)(sizeof(sat->points)/sizeof(sat->points[0]));
  for (PetscInt type = 0; type < num_types; ++type) {
    khash_t(CharacteristicCurvesPointMap) *point_map =
      kh_init(CharacteristicCurvesPointMap);
    PetscInt num_points = sat->num_points[type];
    for (PetscInt i = 0; i < num_points; ++i) {
      int ret;
      khiter_t iter = kh_put(CharacteristicCurvesPointMap, point_map,
                             sat->points[type][i], &ret);
      kh_value(point_map, iter) = i;
    }
    sat->point_maps[type] = point_map;
  }
  PetscFunctionReturn(0);
}

/// Computes the saturation on only the points with the specified indices,
/// according to the type set for each point.
/// @param [in] sat the Saturation inѕtance
/// @param [in] n the number of points on which to compute the saturation
/// @param [in] points the indices of the points
/// @param [in] Sr the residual saturation values ONLY on the points
/// @param [in] Pc the capillary pressure values ONLY on the points
/// @param [out] S the computed saturation values ONLY on the points
/// @param [out] dSdP the computed values of the derivative of saturation w.r.t.
///                   pressure ONLY on the points
/// @param [out] d2SdP2 the computed values of the second derivative of
///                     saturation w.r.t. pressure ONLY on the points
PetscErrorCode SaturationComputeOnPoints(Saturation *sat,
                                         PetscInt n, PetscInt *points,
                                         PetscReal *Sr, PetscReal *Pc,
                                         PetscReal *S, PetscReal *dSdP,
                                         PetscReal *d2SdP2) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // If we haven't built our point sets yet, do so now.
  if (!sat->point_maps[0]) {
    ierr = SaturationBuildPointMaps(sat); CHKERRQ(ierr);
  }

  int num_types = (int)(sizeof(sat->points)/sizeof(sat->points[0]));

  // Loop over the points.
  for (PetscInt i = 0; i < n; ++i) {
    PetscInt i1 = points[i];
    for (int type = 0; type < num_types; ++type) {
      khash_t(CharacteristicCurvesPointMap) *point_map = sat->point_maps[type];
      if (kh_size(point_map) != 0) {
        khiter_t iter = kh_get(CharacteristicCurvesPointMap, point_map, i1);
        if (kh_exist(point_map, iter)) {
          PetscInt j = kh_value(point_map, iter);
          // Compute the saturation and its derivatives on points belonging to
          // this type.
          if (type == SAT_FUNC_GARDNER) {
            PetscReal n = sat->parameters[type][3*j];
            PetscReal m = sat->parameters[type][3*j+1];
            PetscReal alpha = sat->parameters[type][3*j+2];
            PressureSaturation_Gardner(n, m, alpha, Sr[i], Pc[i],
                                       &(S[i]), &(dSdP[i]), &(d2SdP2[i]));
          } else if (type == SAT_FUNC_VAN_GENUCHTEN) {
            PetscReal m = sat->parameters[type][2*j];
            PetscReal alpha = sat->parameters[type][2*j+1];
            PressureSaturation_VanGenuchten(m, alpha, Sr[i], Pc[i],
                                            &(S[i]), &(dSdP[i]), &(d2SdP2[i]));
          } else {
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER,
                    "Invalid saturation model type!");
          }
          break;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/// Creates a new instance of a relative permeability model.
PetscErrorCode RelativePermeabilityCreate(RelativePermeability **rel_perm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = PetscCalloc(sizeof(RelativePermeability), rel_perm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Frees all resources associated with a given Saturation instance.
PetscErrorCode RelativePermeabilityDestroy(RelativePermeability *rel_perm) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  int num_types = (int)(sizeof(rel_perm->points)/sizeof(rel_perm->points[0]));
  for (int type = 0; type < num_types; ++type) {
    RelativePermeabilitySetType(rel_perm, type, 0, NULL, NULL);
  }
  ierr = PetscFree(rel_perm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Sets the type of the relative permeability model used for a give set of
/// points, and its parameters. Point indices and parameters are copied into
/// place.
/// @param [in] rel_perm the RelativePermeability inѕtance
/// @param [in] type the type of the relative permeability model
/// @param [in] num_points the number of points on which the given saturation
///                        model type operates
/// @param [in] points an array of length num_points containing point indices
/// @param [in] parameters an array of length num_params*num_points, with
///                        parameters[num_params*i] containing the first
///                        parameter for the ith point. The number of parameters
///                        depends on the model used.
PetscErrorCode RelativePermeabilitySetType(RelativePermeability *rel_perm,
                                           RelativePermeabilityType type,
                                           PetscInt num_points,
                                           PetscInt *points,
                                           PetscReal *parameters) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  int num_params = 0;
  if (type == REL_PERM_FUNC_IRMAY) {
    num_params = 1;
  } else if (type == REL_PERM_FUNC_MUALEM) {
    num_params = 9;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Invalid relative permeability model type!");
  }

  // Delete the point maps, which are invalidated by this operation.
  int num_types = (int)(sizeof(rel_perm->points)/sizeof(rel_perm->points[0]));
  for (int type = 0; type < num_types; ++type) {
    if (rel_perm->point_maps[type]) {
      khash_t(CharacteristicCurvesPointMap) *point_map = rel_perm->point_maps[type];
      kh_destroy(CharacteristicCurvesPointMap, point_map);
      rel_perm->point_maps[type] = NULL;
    }
  }

  // If we're changing the number of points for this type, free storage.
  if (rel_perm->num_points[type] != num_points) {
    if (rel_perm->points[type]) {
      ierr = PetscFree(rel_perm->points[type]); CHKERRQ(ierr);
      ierr = PetscFree(rel_perm->parameters[type]); CHKERRQ(ierr);
    }
  }

  rel_perm->num_points[type] = num_points;
  ierr = PetscMalloc(num_points*sizeof(PetscInt),
                     &(rel_perm->points[type])); CHKERRQ(ierr);
  memcpy(rel_perm->points[type], points, num_points*sizeof(PetscInt));
  ierr = PetscMalloc(num_params*num_points*sizeof(PetscReal),
                     &(rel_perm->parameters[type])); CHKERRQ(ierr);
  memcpy(rel_perm->parameters[type], parameters,
         num_params*num_points*sizeof(PetscReal));
  PetscFunctionReturn(0);
}

/// Computes the relative permeability for the points assigned to the given
/// type.
/// @param [in] rel_perm the RelativePermeability inѕtance
/// @param [in] Se the effective saturation values on the points
/// @param [out] Kr the computed relative permeability values on the points
/// @param [out] dKrdSe the computed values of the derivative of the relative
///                     permeability w.r.t. effective saturation on the points
PetscErrorCode RelativePermeabilityCompute(RelativePermeability *rel_perm,
                                           PetscReal *Se, PetscReal *Kr,
                                           PetscReal *dKrdSe) {
  PetscFunctionBegin;
  int num_types = (int)(sizeof(rel_perm->points)/sizeof(rel_perm->points[0]));
  for (int type = 0; type < num_types; ++type) {
    PetscInt num_points = rel_perm->num_points[type];
    if (type == REL_PERM_FUNC_IRMAY) {
      for (PetscInt i = 0; i < num_points; ++i) {
        PetscReal m = rel_perm->parameters[type][i];
        RelativePermeability_Irmay(m, Se[i], &(Kr[i]), &(dKrdSe[i]));
      }
    } else if (type == REL_PERM_FUNC_MUALEM) {
      for (PetscInt i = 0; i < num_points; ++i) {
        PetscReal m = rel_perm->parameters[type][9*i];
        PetscReal poly_x0 = rel_perm->parameters[type][9*i+1]; // cubic interp cutoff
        PetscReal poly_x1 = rel_perm->parameters[type][9*i+2]; // cubic interp cutoff
        PetscReal poly_dx = rel_perm->parameters[type][9*i+4]; // cubic interp cutoff
        PetscReal *poly_coeffs = &(rel_perm->parameters[type][9*i+5]); // interp coeffs
        RelativePermeability_Mualem(m, poly_x0, poly_x1, poly_dx, poly_coeffs, Se[i], &(Kr[i]),
                                    &(dKrdSe[i]));
      }
    } else {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER,
              "Invalid relative permeability model type!");
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RelativePermeabilityBuildPointMaps(RelativePermeability *rel_perm) {
  PetscFunctionBegin;
  PetscInt num_types = (PetscInt)(sizeof(rel_perm->points)/sizeof(rel_perm->points[0]));
  for (PetscInt type = 0; type < num_types; ++type) {
    khash_t(CharacteristicCurvesPointMap) *point_map =
      kh_init(CharacteristicCurvesPointMap);
    PetscInt num_points = rel_perm->num_points[type];
    for (PetscInt i = 0; i < num_points; ++i) {
      int ret;
      khiter_t iter = kh_put(CharacteristicCurvesPointMap, point_map,
                             rel_perm->points[type][i], &ret);
      kh_value(point_map, iter) = i;
    }
    rel_perm->point_maps[type] = point_map;
  }
  PetscFunctionReturn(0);
}

/// Computes the relative permeability on only the points with the specified
/// indices, according to the type set for each point.
/// @param [in] rel_perm the RelativePermeability inѕtance
/// @param [in] n the number of points on which to compute the permeability
/// @param [in] points the indices of the points
/// @param [in] Se the effective saturation values ONLY on the points
/// @param [out] Kr the computed relative permeability values ONLY on the points
/// @param [out] dKrdSe the computed values of the derivative of the relative
///                     permeability w.r.t. effective saturation ONLY on the points
PetscErrorCode RelativePermeabilityComputeOnPoints(RelativePermeability *rel_perm,
                                                   PetscInt n, PetscInt *points,
                                                   PetscReal *Se, PetscReal *Kr,
                                                   PetscReal *dKrdSe) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // If we haven't built our point sets yet, do so now.
  if (!rel_perm->point_maps[0]) {
    ierr = RelativePermeabilityBuildPointMaps(rel_perm); CHKERRQ(ierr);
  }

  int num_types = (int)(sizeof(rel_perm->points)/sizeof(rel_perm->points[0]));

  // Loop over the points.
  for (PetscInt i = 0; i < n; ++i) {
    PetscInt i1 = points[i];
    for (int type = 0; type < num_types; ++type) {
      khash_t(CharacteristicCurvesPointMap) *point_map = rel_perm->point_maps[type];
      if (kh_size(point_map) != 0) {
        khiter_t iter = kh_get(CharacteristicCurvesPointMap, point_map, i1);
        if (kh_exist(point_map, iter)) {
          PetscInt j = kh_value(point_map, iter);
          // Compute the relative permeability and its derivative on points
          // belonging to this type.
          if (type == REL_PERM_FUNC_IRMAY) {
            PetscReal m = rel_perm->parameters[type][j];
            RelativePermeability_Irmay(m, Se[i], &(Kr[i]), &(dKrdSe[i]));
          } else if (type == REL_PERM_FUNC_MUALEM) {
            PetscReal m = rel_perm->parameters[type][9*j];
            PetscReal poly_x0 = rel_perm->parameters[type][9*j+1]; // cubic interp cutoff
            PetscReal poly_x1= rel_perm->parameters[type][9*j+2]; // cubic interp cutoff
            PetscReal poly_dx = rel_perm->parameters[type][9*j+4]; // cubic interp cutoff
            PetscReal *poly_coeffs = &(rel_perm->parameters[type][9*j+5]); // interp coeffs
            RelativePermeability_Mualem(m, poly_x0, poly_x1, poly_dx, poly_coeffs, Se[i],
                                        &(Kr[i]), &(dKrdSe[i]));
          } else {
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER,
                    "Invalid relative permeability model type!");
          }
          break;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/// Compute value and derivate of relative permeability using Irmay function
///
/// @param [in] m            parameter for Irmay function
/// @param [in] Se           effective saturation
/// @param [inout] *Kr       value of relative permeability
/// @param [inout] *dKr_dSe  derivate of relative permeability w.r.t. Se
///
/// kr = Se^m  if Se < 1.0
///    = 1     otherwise
///
/// dkr/dSe = m * Se^{m-1}  if Se < 1.0
///         = 0             otherwise
///
void RelativePermeability_Irmay(PetscReal m,PetscReal Se,PetscReal *Kr,
                                PetscReal *dKr_dSe) {
  *Kr = 1.0;
  if (dKr_dSe) *dKr_dSe = 0.0;

  if (Se>=1.0) return;

  *Kr = PetscPowReal(Se,m);
  if(dKr_dSe) *dKr_dSe = PetscPowReal(Se,m-1)*m;
}

/// Compute value and derivate of relative permeability using Mualem function
///
/// @param [in] m            parameter for Mualem function
/// @param [in] Se           effective saturation
/// @param [inout] *Kr       value of relative permeability
/// @param [inout] *dKr_dSe  derivate of relative permeability w.r.t. Se
///
/// kr = Se^{0.5} * [ 1 - (1 - Se^{1/m})^m ]^2     if P < P_ref or Se < 1.0
///    = 1                                         otherwise
///
/// dkr/dSe = 0.5 Se^{-0.5} [ 1 - (1 - Se^{1/m})^m ] +
///           Se^{0.5}      2 * Se^{1/m - 1/} * (1 - Se^{1/m})^{m - 1} * (1 - (1 - Se^{1/m})^m)  if Se < 1.0
///         = 0                                                                                  otherwise
///
void RelativePermeability_Mualem_Unsmoothed(PetscReal m,PetscReal Se,PetscReal *Kr,
				 PetscReal *dKr_dSe) {
  PetscReal Se_one_over_m,tmp;

  *Kr = 1.0;
  if(dKr_dSe) *dKr_dSe = 0.0;

  if (Se>=1.0) return;

  Se_one_over_m = PetscPowReal(Se,1/m);
  tmp = PetscPowReal(1-Se_one_over_m,m);
  (*Kr)  = PetscSqrtReal(Se);
  (*Kr) *= PetscSqr(1-tmp);
  if(dKr_dSe){
    (*dKr_dSe)  = 0.5*(*Kr)/Se;
    (*dKr_dSe) += 2*PetscPowReal(Se,1/m-0.5) * PetscPowReal(1-Se_one_over_m,m-1) * (1-PetscPowReal(1-Se_one_over_m,m));
  }
}

/// Sets up a cubic polynomial interpolation for relative permeability following
/// PFLOTRAN's approach of smoothing relative permeability functions
///
/// @param [in] x1           low value of x
/// @param [in] x2           high value of x
/// @param [inout] *rhs      rhs vector (input) and coefficients (output)
///
///  f(x)  = a0 + a1 * (x-x0)/dx + a2 * ((x-x0)/dx)^2 + a2 * ((x-x0)/dx)^3
///  df_dx =     a1/dx + 2 * a2 * ((x-x0)/dx)/dx + 3 * a3 * ((x-x0)/dx)^2/dx
///
/// Constraints:
/// f(low)      = rel_perm_fn(x)      = f1
/// f(high)     = 1.0                 = f2
/// df_dx(low)  = drv_rel_perm_fn(x)  = df1_dx
/// df_dy(high) = 0.0                 = df2_dx
///
/// Linear system:
/// a0 + a1 * (x1-x0)/dx + a2 * ((x1-x0)/dx)^2   + a2 * ((x1-x0)/dx)^3     = f1
/// a0 + a1 * (x2-x0)/dx + a2 * ((x2-x0)/dx)^2   + a2 * ((x2-x0)/dx)^3     = f2
///      a1 / dx         + 2 * a2 * ((x1-x0)/dx)/dx + 3 * a3 * ((x1-x0)/dx)^2/dx = df1_dx
///      a1 / dx         + 2 * a2 * ((x2-x0)/dx)/dx + 3 * a3 * ((x2-x0)/dx)^2/dx = df2_dx
///
static PetscErrorCode CubicPolynomialSetup(PetscReal x0, PetscReal x1, PetscReal x2, PetscReal dx,
                                           PetscReal rhs[4]) {

  PetscReal xt1 = (x1-x0)/dx, xt2 = (x2-x0)/dx;

  PetscReal xt1_p2 = PetscPowRealInt(xt1,2);
  PetscReal xt2_p2 = PetscPowRealInt(xt2,2);
  PetscReal xt1_p3 = PetscPowRealInt(xt1,3);
  PetscReal xt2_p3 = PetscPowRealInt(xt2,3);

  PetscReal A[16];
  A[ 0] = 1.0   ;  A[ 1] = 1.0   ; A[ 2] = 0.0          ; A[ 3] = 0.0;
  A[ 4] = xt2   ;  A[ 5] = xt1   ; A[ 6] = 1.0       /dx; A[ 7] = 1.0       /dx;
  A[ 8] = xt2_p2;  A[ 9] = xt1_p2; A[10] = 2.0*xt2   /dx; A[11] = 2.0*xt1   /dx;
  A[12] = xt2_p3;  A[13] = xt1_p3; A[14] = 3.0*xt2_p2/dx; A[15] = 3.0*xt1_p2/dx;

  PetscInt n = 4, nrhs = 1;
  PetscInt lda=4, ldb=4;
  PetscInt info; // success/failure from LAPACK
  PetscInt ipiv[n]; // pivot indices

  dgesv_( &n, &nrhs, A, &lda, ipiv, rhs, &ldb, &info );

  if (info > 0) {
    PetscPrintf(PETSC_COMM_WORLD,"");
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"CubicPolynomialSetup failed");
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
///
/// Computes the value and derivative of the cubic polynomial
///
/// @param [in] coeffs   coefficients of a cubic polynomals
/// @param [in] x        value
/// @param [out] *f      function evaluated at value
/// @param [out] *df_dx  derivative of the function evaluated at value = x
///
///  f(x)  = a0 + a1 * (x-x0)/dx + a2 * ((x-x0)/dx)^2 + a2 * ((x-x0)/dx)^3
///  df_dx =     a1/dx + 2 * a2 * ((x-x0)/dx)/dx + 3 * a3 * ((x-x0)/dx)^2/dx
///
static PetscErrorCode CubicPolynomialEvaluate(PetscReal *coeffs, PetscReal x, PetscReal x0, PetscReal dx,
                                              PetscReal *f, PetscReal *df_dx) {

  PetscFunctionBegin;

  PetscReal xt = (x-x0)/dx;

  *f = coeffs[0] + xt * (coeffs[1] + xt * (coeffs[2] + xt * coeffs[3]));
  *df_dx = (coeffs[1] + xt * (2.0 * coeffs[2] + xt * 3.0 * coeffs[3]))/dx;

  PetscFunctionReturn(0);
}

/// Sets up cubic polynomial smoothing for Mualem relative permeability function
///
/// @param [inout] *cc  Charcteristic curve
/// @param [in] ncells  Number of cells
///
/// Computes the coefficients for the cubic polynomial used to smooth the Mualem
/// relative permeability between poly_low and 1.
/// @param [in] m         Mualem parameter
/// @param [in] poly_x0   Value of Se used in variable ransformation for cubic polynomial smoothing
/// @param [in] poly_x1   The minimum value of Se for polynomial smoothing
/// @param [in] poly_x2   The maximum value of Se for polynomial smoothing
/// @param [in] poly_dx   Value of Se used in variable ransformation for cubic polynomial smoothing
/// @param [out] *coeffs  Coefficents of cubic polynomial
PetscErrorCode RelativePermeability_Mualem_GetSmoothingCoeffs(PetscReal m,
                                                              PetscReal poly_x0,
                                                              PetscReal poly_x1,
                                                              PetscReal poly_x2,
                                                              PetscReal poly_dx,
                                                              PetscReal *coeffs) {
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscReal Kr, dKr_dSe;
  RelativePermeability_Mualem_Unsmoothed(m, poly_x1, &Kr, &dKr_dSe);

  coeffs[0] = 1.0;
  coeffs[1] = Kr;
  coeffs[2] = 0.0;
  coeffs[3] = dKr_dSe;

  ierr = CubicPolynomialSetup(poly_x0, poly_x1, poly_x2, poly_dx, coeffs); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/// Computes relative permeability using Mualem function
/// @param [in] m         Mualem parameter
/// @param [in] poly_x0   Value of Se used in variable ransformation for cubic polynomial smoothing
/// @param [in] poly_x1   Value of Se above which polynomial smoothing should be done
/// @param [in] poly_dx   Value of Se used in variable ransformation for cubic polynomial smoothing
/// @param [in] *coeffs   Coefficents of cubic polynomial
/// @param [in] Se        Effective saturation
/// @param [out] *Kr      Relative permeability
/// @param [out] *dKr_dSe Derivative of relative permeability
void RelativePermeability_Mualem(PetscReal m, PetscReal poly_x0, PetscReal poly_x1, PetscReal poly_dx, PetscReal *coeffs, PetscReal Se,PetscReal *Kr,PetscReal *dKr_dSe) {

  if (Se > poly_x1) {
    CubicPolynomialEvaluate(coeffs, Se, poly_x0, poly_dx, Kr, dKr_dSe);
  } else {
    RelativePermeability_Mualem_Unsmoothed(m, Se, Kr, dKr_dSe);
  }
}

/* -------------------------------------------------------------------------- */
/// Compute value and derivates of saturation using Gardner function
///
/// @param [in] n            parameter for Gardner function
/// @param [in] m            parameter for Gardner function
/// @param [in] alpha        parameter for Gardner function
/// @param [in] Sr           residual saturation
/// @param [in] Pc           capillary pressure
/// @param [inout] S         value of saturation
/// @param [inout] *dS_dP    first derivate of saturation w.r.t. pressure
/// @param [inout] *d2S_dP2  second derivate of saturation w.r.t. pressure
///
/// Se = exp(-alpha/m*Pc) if Pc < 0.0
///    = 1                otherwise
///
/// S = (1 - Sr)*Se + Sr
///
/// dSe/dPc = -alpha/m*exp(-alpha/m*Pc-1) if Pc < 0.0
///         = 0                            otherwise
///
/// dS/dP  = dS/dPc * dPc/dP
///        = dSe/dPc * dS/dSe * dPc/dP
///
/// and dS/dSe = 1 - Sr; dPc/dP = -1
///
void PressureSaturation_Gardner(PetscReal n,PetscReal m,PetscReal alpha, PetscReal Sr,
                                PetscReal Pc,PetscReal *S,PetscReal *dS_dP,PetscReal *d2S_dP2) {
  if(Pc < 0) { /* if Pc < 0 then P > Pref and Se = 1 */
    *S = 1;
    if(dS_dP) *dS_dP = 0;
    if(d2S_dP2) *d2S_dP2 =0.0;
  }else{
    PetscReal Se, dSe_dPc;
    Se = PetscExpReal(-alpha*Pc/m);
    *S = (1.0 - Sr)*Se + Sr;
    if(dS_dP) {
      dSe_dPc = -alpha/m*PetscExpReal(-alpha*Pc/m);
      *dS_dP = -dSe_dPc*(1.0 - Sr);
      if (d2S_dP2) {
        PetscReal d2Se_dPc2;
        d2Se_dPc2 = -alpha/m*dSe_dPc;
        *d2S_dP2 = (1.0-Sr)*d2Se_dPc2;
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/// Compute value and derivates of saturation using Van Genuchten function
///
/// @param [in] m            parameter for van Genuchten function
/// @param [in] alpha        parameter for van Genuchten function
/// @param [in] Sr           residual saturation
/// @param [in] Pc           capillary pressure
/// @param [inout] S         value of saturation
/// @param [inout] *dS_dP    first derivate of saturation w.r.t. pressure
/// @param [inout] *d2S_dP2  second derivate of saturation w.r.t. pressure
///
///  Se = [1 + (a * Pc)^(1/(1-m))]^{-m}   if  Pc < 0
///     = 1                               otherwise
///
///  Let n = 1/(1-m)
///
/// dSe/dPc = - [m * n * a * (a*Pc)^n] / denom
/// denom   = (a*Pc) * [ (a*Pc)^n + 1]^{m+1}
///
/// S = (1 - Sr)*Se + Sr
///
/// dSe/dPc = -alpha/m*exp(-alpha/m*Pc-1) if Pc < 0.0
///         = 0                            otherwise
///
/// dS/dP  = dS/dPc * dPc/dP
///        = dSe/dPc * dS/dSe * dPc/dP
///
/// and dS/dSe = 1 - Sr; dPc/dP = -1
///
void PressureSaturation_VanGenuchten(PetscReal m,PetscReal alpha,  PetscReal Sr,
				     PetscReal Pc,PetscReal *S,PetscReal *dS_dP,PetscReal *d2S_dP2) {
  PetscReal pc_alpha,pc_alpha_n,one_plus_pc_alpha_n,n;
  if(Pc <= 0) {
    *S = 1;
    if(dS_dP) *dS_dP = 0;
    if(d2S_dP2) *d2S_dP2 =0.0;
  }else{
    PetscReal Se, dSe_dPc;
    n = 1/(1-m);
    pc_alpha = Pc*alpha;
    pc_alpha_n = PetscPowReal(pc_alpha,n);
    one_plus_pc_alpha_n = 1+pc_alpha_n;
    Se = PetscPowReal(one_plus_pc_alpha_n,-m);
    *S = (1.0 - Sr)*Se + Sr;
    if(dS_dP){
      dSe_dPc = -m*n*alpha*pc_alpha_n/(pc_alpha*PetscPowReal(one_plus_pc_alpha_n,m+1));
      *dS_dP = -dSe_dPc*(1.0 - Sr);
      if (d2S_dP2) {
        PetscReal d2Se_dPc2;
        d2Se_dPc2 = m*n*(pc_alpha_n)*PetscPowReal(one_plus_pc_alpha_n,-m-2.0)*((m*n+1.0)*pc_alpha_n-n+1)/PetscPowReal(Pc,2.0);
        *d2S_dP2 = (1.0-Sr)*d2Se_dPc2;
      }
    }
  }
}
