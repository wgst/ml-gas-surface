#!/bin/bash
#SBATCH -J ace_fin
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --mail-type=END
#SBATCH --mail-user=<wojciech.stark@warwick.ac.uk>


module purge
module load GCCcore/10.2.0
module load Python/3.8.6
source ~/ace_pot2/bin/activate

export PATH="$PATH:/home/chem/msrmnk/2_software/julia-1.10.0/bin"


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
ulimit -s unlimited

which python
which julia



# SPLIT 5
export SPLIT_NO=5

export CUTOFF=5.0
export N_EPOCHS=3000
export WEIGHT_DIAG=2.0
export WEIGHT_SUBDIAG=1.0
export WEIGHT_OFFDIAG=1.0
export MATRIX_BOND_WEIGHT=0.5
export N_REP_INV=2
export N_REP_COV=3
export N_REP_EQU=5
export MAX_ORDER=2
export MAX_DEGREE=6
export N_TRAIN=1320
export N_TEST=330

export DB_EFT=/home/chem/msrmnk/1_calculations/6_ml/3_eft_model/2_h2cu/0_db/h2cu_eft_20240127_full_shfl_up_red.json
export OUTPUT_PATH=model_${SPLIT_NO}/

julia fit_aceds_ac.jl
julia get_aceds_errors_plots.jl
julia get_speeds.jl
