# using Pkg
# Pkg.activate("/home/m/msrmnk2/FrictionProviders/cur")
# Pkg.instantiate()


using NQCDynamics
using NQCDynamics.InitialConditions.QuantisedDiatomic
using DelimitedFiles
using PyCall
using Unitful
using UnitfulAtomic
using Plots
using Statistics
using NNInterfaces
using LinearAlgebra
using JLD2
using StatsBase: var
using Distributions: Normal
using NQCDistributions
using Random: shuffle!

# Importing ASE modules with PyCall by using "pyimport" 
io = pyimport("ase.io")
constraints = pyimport("ase.constraints")
build = pyimport("ase.build")
ase_units = pyimport("ase.units")
mace_calc = pyimport("mace.calculators")

function deltar(r)
    r[:,1]-r[:,2] #difference vector
end

function cm_pos_orient(positions)
    x_cm = []
    y_cm = []
    alpha = []
    beta = []
    # we compute the positions and orientations associted with CM
    #-------------------------------------------------------------
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
    # We compute the lattice vectors to locate H2 molecule to figure out "z0"
    a1 = atoms.cell[1]
    a2 = atoms.cell[2]
    h2_molecule = build.molecule("H2")
    h2_molecule.rotate(90, (1,0,0))
    r_dist = h2_molecule.positions[1,:] - h2_molecule.positions[2,:]
    h2_molecule.rotate(r_dist, a1+a2)
    adsorb_position = (a1+a2)/2
    return h2_molecule, adsorb_position
end

function mace_model(model_path, cur_atoms)
    calculator = mace_calc.MACECalculator(model_path=model_path, device="cpu", default_dtype="float32") # or "cuda"
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end

function get_constraints_ase(ase_slab, range)
    mask = zeros(length(ase_slab)) #constrain 2 bottom layers
	mask[range] .= 1
    constraint = constraints.FixAtoms(mask=mask)
    return constraint
end

"""
--------------------------------------------------------------------------------------------------------------------------
 This script generates the initial positions and velocities(r,v) associated with a specific pair of quantum numbers (v,J) 
-------------------------------------------------------------------------------------------------------------------------
"""

#  --- External file block ---
# output file path
output_f=ENV["H2CU_ML_INIT_COND_OUT"]
mkpath(output_f)
# path to find the Cu coordinate taken from training data
input_f=ENV["H2CU_ML_INIT_COND_IN"]
# ML model path
ml_model_f=ENV["PES_MODEL_PATH"]
ml_model_ver=ENV["PES_MODEL_VER"]


# settings
surfaces = ["cu110"]
energies_translational = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4] # state-to-state 1


Ts = [300] 
heights = [7.0] # AA
dof = 3
vjs = [[1,1]]
samples_num = 50000


# -------- Main block -----
for surface in surfaces
    if surface=="cu211"
        n_atoms = 74
    else
        n_atoms = 56
    end
    temp = Ts[1]
    surfaces_mc = io.read("$(ENV["SURF_FOLDER"])/$(surface)/output/nvt_$(temp)K.traj@4001:")
    shuffle!(surfaces_mc)
    height = heights[1]
    
    # create template slab H2 + Cu
    ase_slab = surfaces_mc[1].copy()
    mol_h2, ads_position = create_adsorbate_h2(ase_slab)
    build.add_adsorbate(ase_slab,mol_h2,height,position=(ads_position[1],ads_position[2]))
    cur_atoms = mol_h2.copy()
    cur_atoms.cell = ase_slab.cell
    xatoms, xpositions, xcell = NQCDynamics.convert_from_ase_atoms(ase_slab) # We transform cur_atoms to Julia object to take "cell"
    set_periodicity!(xcell,[true,true,true])

    println(" Initialize ...  ") # We combine models into average potential
    model_h2 = mace_model(ml_model_f, cur_atoms)
    for vj in vjs
        for e_tran in energies_translational
            v  = vj[1]
            J  = vj[2]
            z  = height + mean(ase_slab.positions[n_atoms-10:n_atoms-2,dof]) #; one H atom      # Height [Å]
            cur_folder_name = "Et$(e_tran)_v$(v)_J$(J)"
            cur_folder_path = "$(output_f)/$(surface)/T$(temp)K/$(ml_model_ver)/$(cur_folder_name)/"
            mkpath(cur_folder_path)
            
            # Distribution initialized at a specific ro-vibrational quantum (v,J) state:
            #---------------------------------------------------------------------------
            println(" New translational energy: $(e_tran)")
            println("$(cur_folder_path)traj_$(cur_folder_name).xyz")
            atoms_h2 = NQCDynamics.Atoms([:H,:H]) 
            sim = Simulation{Classical}(atoms_h2, model_h2, cell=xcell)
            configs = QuantisedDiatomic.generate_configurations(sim, v, J; samples=samples_num, translational_energy=e_tran*u"eV", height=ang_to_au(z)) 

            # H2 velocities and positions
            v_h2 = first.(configs)
            r_h2 = last.(configs)

            # Re build the position and velocities for the whole system Cu56H2
            velos = [zeros(dof,n_atoms) for i=1:length(r_h2)]
            f_positions = [zeros(dof,n_atoms) for i=1:length(r_h2)]

            # boltz_dist = VelocityBoltzmann(temp*u"K", xatoms.masses[19:n_atoms-2],(3,n_atoms-20)) # only for Cu
            for i in 1:length(r_h2)
                surf_atoms, surf_positions, surf_cell = NQCDynamics.convert_from_ase_atoms(surfaces_mc[i])
                f_positions[i][:,1:n_atoms-2] .= surf_positions #[:,1:n_atoms-2]
                f_positions[i][:,n_atoms-1:n_atoms] .= r_h2[i]  

                velos[i][:,1:n_atoms-2] .= austrip.(transpose(surfaces_mc[i].get_velocities().*ase_units.fs)*u"Å/fs")
                velos[i][:,n_atoms-1:n_atoms] .= v_h2[i]
            end

            println("Creating distribution ...")
            dist = DynamicalDistribution(velos, f_positions, (dof,n_atoms))

            #save output 
            atoms_print = NQCDynamics.convert_to_ase_atoms(xatoms, f_positions, xcell)
            io.write("$(cur_folder_path)traj_$(cur_folder_name).xyz",atoms_print)

            println("printing initial conditions ...")
            JLD2.save("$(cur_folder_path)distr_$(cur_folder_name).jld2",Dict("dist"=>dist,"atoms"=>xatoms,"cell"=>xcell))

            # Plotting initial conditions
            #-----------------------------
            println("Plotting ...")
            x_cm, y_cm, alpha, beta = cm_pos_orient(r_h2)

            p1 = scatter(x_cm, y_cm, label="CM position", xlabel="xCM position [bohr]", ylabel="yCM position [bohr]", framestyle=:box, dpi=300)
            p2 = scatter(alpha, beta, color="red", label="orientation", xlabel="azimuthal angle \\phi [Deg]", ylabel="polar angle \\theta [Deg]", framestyle=:box, dpi=300)
            p = plot(p1, p2, layout=(2,1), size=(400,600), legend=false)
            path_n = "$(cur_folder_path)distr_$(cur_folder_name).png"
            savefig(p, path_n)
            
            println("end ...")
        end
    end

end


