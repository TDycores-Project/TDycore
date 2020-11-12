TARGETS = elm

ifdef codecov
  MYFLAGS += -fprofile-arcs -ftest-coverage
  LIBS    += -lgcov
endif

# These flags are supplemental to the PETSc flags
CFLAGS   = ${LIBS}
FFLAGS   = ${LIBS}
CPPFLAGS = ${MYFLAGS}
FPPFLAGS = ${MYFLAGS}

ALL: $(TARGETS)
clean::
	-@$(RM) $(TARGETS) *.gcov *.regression *.old *.stdout *.bin *.bin.info

topdir := $(shell cd ../.. && pwd)
TDYCORE_DIR ?= $(topdir)
include $(TDYCORE_DIR)/lib/tdycore/conf/variables
include $(TDYCORE_DIR)/lib/tdycore/conf/rules

#shr_infnan_mod.o \
#shr_assert_mod.o \
#shr_mpi_mod.o \

elm_obj = \
shr_kind_mod.o \
shr_log_mod.o \
shr_sys_mod.o \
clm_varctrl.o \
clm_varpar.o \
abortutils.o \
decompMod.o \
spmdMod.o \
shr_const_mod.o \
landunit_varcon.o \
EMI_Atm2LndType_Constants.o \
EMI_CNCarbonStateType_Constants.o \
EMI_CanopyStateType_Constants.o \
EMI_ChemStateType_Constants.o \
EMI_ColumnType_Constants.o \
EMI_EnergyFluxType_Constants.o \
EMI_Filter_Constants.o \
EMI_Landunit_Constants.o \
EMI_SoilHydrologyType_Constants.o \
EMI_SoilStateType_Constants.o \
EMI_TemperatureType_Constants.o \
EMI_WaterFluxType_Constants.o \
EMI_WaterStateType_Constants.o \
ExternalModelConstants.o \
elm_tdycore_interface_data.o \
mapping.o \
EMI_DataDimensionMod.o \
EMI_Atm2LndType_DataMod.o \
EMI_CNCarbonStateType_DataMod.o \
EMI_CanopyStateType_DataMod.o \
EMI_ChemStateType_DataMod.o \
EMI_EnergyFluxType_DataMod.o \
EMI_SoilHydrologyType_DataMod.o \
EMI_SoilStateType_DataMod.o \
EMI_TemperatureType_DataMod.o \
EMI_WaterFluxType_DataMod.o \
EMI_WaterStateType_DataMod.o \
EMI_DataMod.o \
clm_varcon.o \
ExternalModelBaseType.o \
ExternalModelTDycore.o \
elm.o

build:
	@cd ../; make

elm: $(elm_obj)
	$(FLINKER) -o elm $(elm_obj) $(TDYCORE_LIB) $(LIBS)

