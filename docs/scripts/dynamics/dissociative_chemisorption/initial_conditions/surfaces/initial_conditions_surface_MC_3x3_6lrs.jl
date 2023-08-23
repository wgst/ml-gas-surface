using NQCDynamics
using NQCDynamics.InitialConditions.QuantisedDiatomic
using NQCDynamics.InitialConditions
using DelimitedFiles
using PyCall
using Unitful
using UnitfulAtomic
using Statistics
using NNInterfaces
using LinearAlgebra
using JLD2
using StatsBase: var
using Distributions: Normal
using NQCDistributions
using Random
using Plots

# Importing Python modules with PyCall
ase = pyimport("ase")
io = pyimport("ase.io")
view = pyimport("ase.visualize")
optimizer = pyimport("ase.optimize")
constraints = pyimport("ase.constraints")
build = pyimport("ase.build")
spk_utils = pyimport("schnetpack.utils")
spk_interfaces = pyimport("schnetpack.interfaces")

"""
Function for creating ASE object from SchNet model
"""
function schnet_model_pes(model_path, cur_atoms)
    spk_model = spk_utils.load_model(model_path;map_location="cpu")
    model_args = spk_utils.read_from_json("$(model_path)/args.json")
    environment_provider = spk_utils.script_utils.settings.get_environment_provider(model_args,device="cpu")
    calculator = spk_interfaces.SpkCalculator(spk_model, energy="energy", forces="forces", environment_provider=environment_provider)
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end

"""
 This script..... 2022  -H2 on Cu-
"""
# -----  Initial block -----------------------
# Initial positions are take from training data: 
icoords_f = ENV["INPUT_STRUCTURE_FOLDER"]
# Output_file path:
output_f = ENV["OUTPUT_FOLDER"]
# Machine learning model path
ml_model_f = ENV["H2CU_ML_MODELS"]

#Settings
surfaces = ["cu111", "cu100", "cu110", "cu211"]
temperature_set = 925 # K
n_atoms = 56
n_atoms_constr = 18 # Number of frozen atoms 

# MC settings
Δ = Dict([(:Cu,0.4)])
passes = 200

# -------- Main block -----
for surface in surfaces
    cur_icoords = "$(icoords_f)$(surface)_$(temperature_set)K.in"
    cur_output = "$(output_f)$(surface)/$(temperature_set)K/"
    mkpath(cur_output)

    ase_slab = io.read(cur_icoords)

    model = schnet_model_pes(ml_model_f, ase_slab)

    atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_slab)
    set_periodicity!(cell,[true,true,true])
    vectors=cell.vectors

    println("Initialize ...  ")
    sim = Simulation{Classical}(atoms, model,cell=cell,temperature=temperature_set*u"K")


    println("Runing MC simulation ....")

    output = MetropolisHastings.run_monte_carlo_sampling(sim, positions, Δ, passes; fix=collect(1:n_atoms_constr))
    @show output.acceptance

    println("Ending first MC...")

    #--- Analisis block --------
    f_positions=shuffle!(output.R)
    println("Number of total configuration ...")
    println(length(f_positions))

    # Printing fcc chain
    atoms_print = NQCDynamics.convert_to_ase_atoms(atoms,f_positions,cell)
    io.write("$(cur_output)mc_chain_$(surface).xyz",atoms_print)
end
