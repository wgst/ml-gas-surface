#!/bin/bash
#SBATCH --job-name=reann_init_925K_1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
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

# CALCULATION DETAILS
export H2CU_ML_INIT_COND_IN=/home/m/msrmnk2/1_calculations/1_ml/12_mdef_state_to_state/2_initial_conditions/0_input_structures
export H2CU_ML_INIT_COND_OUT=output
export PES_MODEL_VER=1
export PES_MODEL_PATH=/home/m/msrmnk2/1_calculations/1_ml/2_training/2_h2cu/31_MACE/10_mace_0_2_0/5_4_ad_float32_final2_FINAL_opt_co5_no_outliers/1/MACE_model_swa.model
export SURF_FOLDER=/home/m/msrmnk2/1_calculations/1_ml/12_mdef_state_to_state/2_initial_conditions/1_surface_md/1_langevin/2_mace/300K
echo $PES_MODEL_PATH
echo $SURF_FILE_PATH

julia initial_conditions_300K.jl






