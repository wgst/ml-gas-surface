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
# module load GCCcore/13.3.0
# module load Python/3.11.3
source ~/ace_pot2/bin/activate

export PATH="$PATH:/home/chem/msrmnk/2_software/julia-1.10.0/bin"
export JULIA_NUM_THREADS=48

export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export MKL_DYNAMIC=FALSE
ulimit -s unlimited


export ACEpot_DATABASE="/home/chem/msrmnk/1_calculations/6_ml/3_eft_model/3_h2cu_ldfa/1_ml_ace_pot/1_db/h2cu_20240205_density.xyz"

##################
export ACEpot_VER=1
##################
# ACE SETTINGS
export ACEpot_CORRELATION_ORDER=3
export ACEpot_POLYNOMIAL_DEGREE=3
export ACEpot_CUTOFF=4
export ACEpot_E_WEIGHT=1
# MODEL NAME
export ACEpot_MODEL_FOLDER=n${ACEpot_CORRELATION_ORDER}_d${ACEpot_POLYNOMIAL_DEGREE}_co${ACEpot_CUTOFF}_${ACEpot_VER}
# RUN THE SCRIPT
# julia acepot_fit.jl
# GET REFERENCE VALUES AND CALCULATE PREDICTIONS
export ACEpot_OPTION_EVAL="test"
julia acepot_calc_results_test.jl
export ACEpot_OPTION_EVAL="train"
julia acepot_calc_results_test.jl


##################
export ACEpot_VER=2
##################
# ACE SETTINGS
export ACEpot_CORRELATION_ORDER=3
export ACEpot_POLYNOMIAL_DEGREE=3
export ACEpot_CUTOFF=4
export ACEpot_E_WEIGHT=1
# MODEL NAME
export ACEpot_MODEL_FOLDER=n${ACEpot_CORRELATION_ORDER}_d${ACEpot_POLYNOMIAL_DEGREE}_co${ACEpot_CUTOFF}_${ACEpot_VER}
# RUN THE SCRIPT
julia acepot_fit.jl
# GET REFERENCE VALUES AND CALCULATE PREDICTIONS
export ACEpot_OPTION_EVAL="test"
julia acepot_calc_results_test.jl
export ACEpot_OPTION_EVAL="train"
julia acepot_calc_results_test.jl


##################
export ACEpot_VER=3
##################
# ACE SETTINGS
export ACEpot_CORRELATION_ORDER=3
export ACEpot_POLYNOMIAL_DEGREE=3
export ACEpot_CUTOFF=4
export ACEpot_E_WEIGHT=1
# MODEL NAME
export ACEpot_MODEL_FOLDER=n${ACEpot_CORRELATION_ORDER}_d${ACEpot_POLYNOMIAL_DEGREE}_co${ACEpot_CUTOFF}_${ACEpot_VER}
# RUN THE SCRIPT
julia acepot_fit.jl
# GET REFERENCE VALUES AND CALCULATE PREDICTIONS
export ACEpot_OPTION_EVAL="test"
julia acepot_calc_results_test.jl
export ACEpot_OPTION_EVAL="train"
julia acepot_calc_results_test.jl



##################
export ACEpot_VER=4
##################
# ACE SETTINGS
export ACEpot_CORRELATION_ORDER=3
export ACEpot_POLYNOMIAL_DEGREE=3
export ACEpot_CUTOFF=4
export ACEpot_E_WEIGHT=1
# MODEL NAME
export ACEpot_MODEL_FOLDER=n${ACEpot_CORRELATION_ORDER}_d${ACEpot_POLYNOMIAL_DEGREE}_co${ACEpot_CUTOFF}_${ACEpot_VER}
# RUN THE SCRIPT
julia acepot_fit.jl
# GET REFERENCE VALUES AND CALCULATE PREDICTIONS
export ACEpot_OPTION_EVAL="test"
julia acepot_calc_results_test.jl
export ACEpot_OPTION_EVAL="train"
julia acepot_calc_results_test.jl



##################
export ACEpot_VER=5
##################
# ACE SETTINGS
export ACEpot_CORRELATION_ORDER=3
export ACEpot_POLYNOMIAL_DEGREE=3
export ACEpot_CUTOFF=4
export ACEpot_E_WEIGHT=1
# MODEL NAME
export ACEpot_MODEL_FOLDER=n${ACEpot_CORRELATION_ORDER}_d${ACEpot_POLYNOMIAL_DEGREE}_co${ACEpot_CUTOFF}_${ACEpot_VER}
# RUN THE SCRIPT
julia acepot_fit.jl
# GET REFERENCE VALUES AND CALCULATE PREDICTIONS
export ACEpot_OPTION_EVAL="test"
julia acepot_calc_results_test.jl
export ACEpot_OPTION_EVAL="train"
julia acepot_calc_results_test.jl

