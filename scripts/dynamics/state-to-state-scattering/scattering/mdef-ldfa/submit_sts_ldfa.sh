#!/bin/bash
#SBATCH --job-name=mace_sts
#SBATCH --time=48:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3850
#SBATCH --account=su007-rjm
#SBATCH --mail-type=END
#SBATCH --mail-user=<wojciech.stark@warwick.ac.uk>


module purge
module load GCCcore/11.3.0 Python/3.10.4
module load GCC/11.3.0 OpenMPI/4.1.4
module load PyTorch/1.12.1-CUDA-11.7.0
source ~/mace_020/bin/activate
which python


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
ulimit -s unlimited


export INIT_COND_FOLDER=/home/m/msrmnk2/1_calculations/1_ml/12_mdef_state_to_state/2_initial_conditions
# export MODEL_FOLDER=/home/m/msrmnk2/1_calculations/1_ml/2_training/2_h2cu

#### SETTINGS ####
export SET_SURFACE=cu111
export SET_TEMPERATURE=300
export TRAJ_START=1
export TRAJ_END=10000
# QUANTUM NUMBERS V AND J
export SET_QN_V=2
export SET_QN_J=1
#### PATHS ####
export PES_MODEL_VER=1
export PES_MODEL_PATH=/home/m/msrmnk2/1_calculations/1_ml/2_training/2_h2cu/31_MACE/10_mace_0_2_0/5_4_ad_float32_final2_FINAL_opt_co5_no_outliers/${PES_MODEL_VER}/MACE_model_swa.model
export EFT_MODEL_PATH=/home/m/msrmnk2/1_calculations/1_ml/12_mdef_state_to_state/1_models/1_ldfa/n3_d3_co4_4/h2cu_ace.json
export H2CU_ML_INPUT_STR=${INIT_COND_FOLDER}/0_input_structures
export H2CU_ML_INIT_COND=${INIT_COND_FOLDER}/2_h2_surface/2_mace_100k/output/${SET_SURFACE}/T${SET_TEMPERATURE}K/${PES_MODEL_VER}
export H2CU_OUTPUTS=output


# energies_translational = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] # H2

echo $TRAJ_START
echo $TRAJ_END

############# 1 #############
export SET_E_TRANSL=0.05
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 2 #############
export SET_E_TRANSL=0.1
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 3 #############
export SET_E_TRANSL=0.15
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 4 #############
export SET_E_TRANSL=0.2
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 5 #############
export SET_E_TRANSL=0.25
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 6 #############
export SET_E_TRANSL=0.3
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 7 #############
export SET_E_TRANSL=0.35
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 8 #############
export SET_E_TRANSL=0.4
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 9 #############
export SET_E_TRANSL=0.45
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl

############# 10 #############
export SET_E_TRANSL=0.5
echo $SET_SURFACE
echo $SET_TEMPERATURE
echo $SET_QN_V
echo $SET_QN_J
echo $SET_E_TRANSL
echo $PES_MODEL_VER
echo $PES_MODEL_PATH
echo $EFT_MODEL_PATH
echo $H2CU_ML_INIT_COND
julia mdef_ldfa_sts_mace.jl
# julia get_vjs.jl
