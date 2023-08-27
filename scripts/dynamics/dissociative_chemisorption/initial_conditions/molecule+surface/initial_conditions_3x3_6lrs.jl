using NQCDynamics
using NQCDynamics.InitialConditions.QuantisedDiatomic
using DelimitedFiles
using PyCall
using Unitful
using UnitfulAtomic
using Plots
using Statistics
using LinearAlgebra
using JLD2
using StatsBase: var
using Distributions: Normal
using NQCDistributions

# Importing Python modules with PyCall
io = pyimport("ase.io")
db = pyimport("ase.db")
build = pyimport("ase.build")
constraints = pyimport("ase.constraints")
spk_utils = pyimport("schnetpack.utils")
spk_interfaces = pyimport("schnetpack.interfaces")

function deltar(r)
    r[:,1]-r[:,2] # Difference vector
end

function cm_pos_orient(positions)
    x_cm = []
    y_cm = []
    alpha = []
    beta = []
    # We compute the positions and orientations associted with CM
    for (n,e) in enumerate(positions)
        push!(x_cm,mean(positions[n][1,:]))              # x_cm = (xh1+xh2)/2   !only valid for diatomic homonuclear molecules
        push!(y_cm,mean(positions[n][2,:]))              # y_cm = (yh1+yh2)/2   !only valid for diatomic homonuclear molecules

        b = acos((deltar(positions[n])[3]/norm(deltar(positions[n]))))    # acos(dr_z/dr)  "thetha"; polar angle [0,pi]
        push!(beta,rad2deg(b))
        
        a = atan(deltar(positions[n])[2]/deltar(positions[n])[1])          #atan(dr_y/dr_x) "phi"; azimuthal angle [0,2pi]
        push!(alpha,rad2deg(a))
    end
    return x_cm, y_cm, alpha, beta
end

function create_adsorbate_h2(atoms)
    a1 = atoms.cell[1]
    a2 = atoms.cell[2]
    h2_molecule = build.molecule("H2")
    h2_molecule.rotate(90, (1,0,0))
    r_dist = h2_molecule.positions[1,:] - h2_molecule.positions[2,:]
    h2_molecule.rotate(r_dist, a1+a2)
    adsorb_position = (a1+a2)/2
    return h2_molecule, adsorb_position
end

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
--------------------------------------------------------------------------------------------------------------------------
 This script generates the initial positions and velocities(r,v) associated with a specific pair of quantum numbers (v,J) 
-------------------------------------------------------------------------------------------------------------------------
"""

model_folder = ENV["MODEL_FOLDER"]

output_f=ENV["H2CU_ML_INIT_COND"]*"/output/$(model_folder)/" # Output file path
mkpath(output_f)
input_f=ENV["H2CU_ML_INIT_COND"]*"/inputs/" # Path to find the Cu coordinate taken e.g. from training data
ml_model_f="$(ENV["H2CU_ML_MODELS"])$(model_folder)" # ML model path

# SETTINGS
surfaces = ["cu111", "cu100", "cu110", "cu211"]
es_tran = [0.200, 0.400, 0.500, 0.600, 0.750, 0.850] # v0
#es_tran = [0.200, 0.300, 0.400, 0.500, 0.600, 0.750] # v1
Ts = [925] # Surface temperature / K
heights = [7.0] # Ang
dof = 3
vjs = [[0,1],[1,1]]
samples_num = 10000
n_atoms = 56
n_atoms_constr = 18 # Number of frozen atoms 

# -------- Main block -----
for surface in surfaces
    temp = Ts[1]
    surfaces_mc = io.read("$(ENV["SURF_FILE_PATH"])$(surface)/$(temp)K/mc_chain_$(surface).xyz@:")
    atoms_all = []
    height = heights[1]
    
    # CREATE AN INITIAL H2 + Cu STRUCTURE
    ase_slab = io.read("$(input_f)$(surface)_$(temp)K.in")
    mol_h2, ads_position = create_adsorbate_h2(ase_slab)
    build.add_adsorbate(ase_slab,mol_h2,height,position=(ads_position[1],ads_position[2]))
    cur_atoms = mol_h2.copy()
    cur_atoms.cell = ase_slab.cell
    xatoms, xpositions, xcell = NQCDynamics.convert_from_ase_atoms(ase_slab) # We transform cur_atoms to Julia object to take "cell"
    set_periodicity!(xcell,[true,true,true])

    println("Initialize...")
    model_h2 = schnet_model_pes(ml_model_f, cur_atoms)
    for vj in vjs
        for e_tran in es_tran
            v  = vj[1]
            J  = vj[2]
            z  = height + mean(ase_slab.positions[n_atoms-10:n_atoms-2,dof]) #; one H atom      # Height [Å]
            cur_folder_name = "Et$(e_tran)_v$(v)_J$(J)"
            cur_folder_path = "$(output_f)$(surface)/T$(temp)K/$(cur_folder_name)/"
            mkpath(cur_folder_path)

            # DISTRIBUTION INITIALIZEZD AT A SPECIFIC RO-VIBRATIONAL QUANTUM (v,J) STATE:
            println("New translational energy: $(e_tran).")
            println("$(cur_folder_path)traj_$(cur_folder_name).xyz")
            atoms_h2 = NQCDynamics.Atoms([:H,:H]) 
            sim = Simulation{Classical}(atoms_h2, model_h2, cell=xcell)
            configs = QuantisedDiatomic.generate_configurations(sim, v, J; samples=samples_num, translational_energy=e_tran*u"eV", height=ang_to_au(z)) 

            # GET VELOSITIES AND POSITIONS
            velos = [zeros(dof,n_atoms) for i=1:length(r_h2)]
            f_positions = [zeros(dof,n_atoms) for i=1:length(r_h2)]

            v_h2 = first.(configs) # H2 velocities
            r_h2 = last.(configs) # H2 positions
            boltz_dist = VelocityBoltzmann(temp*u"K", xatoms.masses[n_atoms_constr+1:n_atoms-2],(3,n_atoms-(n_atoms_constr+2))) # only for Cu

            for i in 1:length(r_h2)
                surf_atoms, surf_positions, surf_cell = NQCDynamics.convert_from_ase_atoms(surfaces_mc[i])
                f_positions[i][:,1:n_atoms-2] .= surf_positions
                f_positions[i][:,n_atoms-1:n_atoms] .= r_h2[i]  

                velos[i][:,n_atoms_constr+1:n_atoms-2] .= rand(boltz_dist)
                velos[i][:,n_atoms-1:n_atoms] .= v_h2[i]
            end

            println("Creating distribution...")
            dist = DynamicalDistribution(velos, f_positions, (dof,n_atoms))

            # SAVE INITIAL CONDITIONS
            atoms_print = NQCDynamics.convert_to_ase_atoms(xatoms, f_positions, xcell)
            io.write("$(cur_folder_path)traj_$(cur_folder_name).xyz",atoms_print)
            append!(atoms_all, atoms_print)

            println("Printing initial conditions...")
            JLD2.save("$(cur_folder_path)distr_$(cur_folder_name).jld2",Dict("dist"=>dist,"atoms"=>xatoms,"cell"=>xcell))

            # PLOTTING INITIAL CONDITIONS
            println("Plotting...")
            x_cm, y_cm, alpha, beta = cm_pos_orient(r_h2)

            p1 = scatter(x_cm, y_cm, label="CM position", xlabel="xCM position [bohr]", ylabel="yCM position [bohr]", framestyle=:box, dpi=300)
            p2 = scatter(alpha, beta, color="red", label="orientation", xlabel="azimuthal angle \\phi [Deg]", ylabel="polar angle \\theta [Deg]", framestyle=:box, dpi=300)
            p = plot(p1, p2, layout=(2,1), size=(400,600), legend=false)
            path_n = "$(cur_folder_path)distr_$(cur_folder_name).png"
            savefig(p, path_n)
        end
    end
    io.write("$(output_f)$(surface)/$(surface)_v$(vjs[1][1])_J$(vjs[1][2])_traj_all_T$(temp).traj", atoms_all)

end


