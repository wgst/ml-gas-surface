using Distributed, SlurmClusterManager
addprocs(SlurmManager())

@everywhere begin
    using Pkg
    Pkg.activate("/home/m/msrmnk2/FrictionProvidersLDFA/cur")
    Pkg.instantiate()
end

@everywhere begin

    # using NNInterfaces
    using NQCDynamics
    using NQCDynamics.InitialConditions.QuantisedDiatomic
    using PyCall
    using DelimitedFiles
    using Unitful
    using UnitfulAtomic
    using SciMLBase
    using LinearAlgebra
    using Statistics
    using JLD2
    using FrictionProviders
    using ASE
    using JuLIP
    using ACE1

    # Importing ASE modules with PyCall by using "pyimport" 
    io = pyimport("ase.io")
    constraints = pyimport("ase.constraints")
    mace_calc = pyimport("mace.calculators")
    db = pyimport("ase.db")

end # @everywhere

@everywhere begin
    """
    Trajectory terminator if H2 is above 'scat_cutoff' or if the distance between H2 is more than 'dist_cutoff'
    """
    mutable struct TrajectoryTerminator
        h2_indices
        scat_cutoff
        dist_cutoff
        slab_range_dif
        slab_ranges
        n_atoms_layer
    end
    function (term::TrajectoryTerminator)(u, t, integrator)::Bool
        R = NQCDynamics.get_positions(u)
        com_h2_z = minimum(R[3,term.h2_indices[1]:term.h2_indices[2]])
        top_surface_avg_z = sum(R[3,end-Int(term.n_atoms_layer)-1:end-2])/term.n_atoms_layer

        if (com_h2_z - top_surface_avg_z) > term.scat_cutoff # [scat_cutoff] Ang above the surface
            return true
        elseif norm(R[:,term.h2_indices[1]] .- R[:,term.h2_indices[2]]) > term.dist_cutoff # at least [dist_cutoff] Ang between H2 = reaction
            return true
        else
            return false
        end
    end

    """
    Function to compute H-H distance atoms 55 and 56
    """
    function dh2(p) 
        norm(p[:,end].-p[:,end-1])  
    end

    """
    Function for removing surface (54 atoms)
    """
    # function remove_surface(ase_atoms)
    #     n_a = length(ase_atoms)
    #     for i in 1:n_a-2 # remove surface 
    #         ase_atoms.pop(0)
    #     end
    #     return ase_atoms
    # end

    function get_constraints_ase(ase_atoms, range)
        mask = zeros(length(ase_atoms)) #constrain 2 bottom layers
        mask[range] .= 1
        constraint = constraints.FixAtoms(mask=mask)
        return constraint
    end

    """
    Function for creating ASE object from SchNet model
    """
    function mace_model(model_path, cur_atoms)
        calculator = mace_calc.MACECalculator(model_path=model_path, device="cpu", default_dtype="float32") # or "cuda"
        cur_atoms.set_calculator(calculator)
        model = AdiabaticASEModel(cur_atoms)

        return model
    end

    function ace_model(model_path, cur_atoms) # cur_atoms has to be JuLIP atoms object
        IP = ACE1.read_dict(load_dict(model_path)["IP"])
        JuLIP.set_calculator!(cur_atoms, IP)
        model = AdiabaticModels.JuLIPModel(cur_atoms)
        
        return model
    end
    

    function get_modes(atoms,friction_atoms_id)
        """Calculates required transformation matrix to convert diatomic
        friction tensor to internal coordinates and normalises the resulting array
        """
        ndim = length(friction_atoms_id)*3
        modes = zeros(ndim,ndim)
        pos1 = atoms.positions[friction_atoms_id[1],:]
        pos2 = atoms.positions[friction_atoms_id[2],:]
        x1 = pos1[1]
        y1 = pos1[2]
        z1 = pos1[3]
        x2 = pos2[1]
        y2 = pos2[2]
        z2 = pos2[3]
        m1 = atoms.get_masses()[friction_atoms_id[1]]
        m2 = atoms.get_masses()[friction_atoms_id[2]]
        mt = m1 + m2
        mr1 = m1/mt
        mr2 = m2/mt
        r = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        r1 = sqrt((x2-x1)^2 + (y2-y1)^2)
        #mode 1 - r
        modes[1,1] = (x1-x2)/r
        modes[2,1] = (y1-y2)/r
        modes[3,1] = (z1-z2)/r
        modes[4,1] = (x2-x1)/r
        modes[5,1] = (y2-y1)/r
        modes[6,1] = (z2-z1)/r
        #mode 2  - theta - polar angle
        modes[1,2] = ((x2-x1)*(z2-z1))/((r^2)*r1)
        modes[2,2] = ((y2-y1)*(z2-z1))/((r^2)*r1)
        modes[3,2] = -r1/(r^2)
        modes[4,2] = ((x1-x2)*(z2-z1))/((r^2)*r1)
        modes[5,2] = ((y1-y2)*(z2-z1))/((r^2)*r1)
        modes[6,2] = r1/(r^2)
        #mode 3 - phi - azimuthal angle
        modes[1,3] = (y1-y2)/(r1^2)
        modes[2,3] = (x2-x1)/(r1^2)
        modes[3,3] = 0
        modes[4,3] = (y2-y1)/(r1^2)
        modes[5,3] = (x1-x2)/(r1^2)
        modes[6,3] = 0
        # X, Y, Z
        modes[:,4] = [mr1,0.,0.,mr2,0.,0.] # mode 4 is the x translation
        modes[:,5] = [0.,mr1,0.,0.,mr2,0.] # mode 5 is the y translation
        modes[:,6] = [0.,0.,mr1,0.,0.,mr2] # mode 6 is the z translation
    
        #normalisation to conserve trace between friction obtained and received tensor 
        for i in 1:6
            modes[:,i] /= norm(modes[:,i])
        end
    
        return modes
    end

    function kinetic_energy_ase(cur_atoms, calculator)
        cur_atoms.set_calculator(calculator)
        return uconvert(u"eV",(cur_atoms.get_kinetic_energy())*((u"u")*((u"a0_au*Eh_au/ħ_au")^2)))
    end
    
    
    function potential_energy(cur_atoms, calculator)
        cur_atoms.set_calculator(calculator)
        return (cur_atoms.get_potential_energy())*u"eV"
    end

    function get_VCOM(m1, m2, V1::Vector, V2::Vector) # function from Nils
        MGES = m1 + m2
        V_COM = 1/MGES * (m1 * V1 + m2 * V2)
        return V_COM
    end

    function get_proj_COM_Ekin(m1, m2, V1::Vector, V2::Vector) # function from Nils
        M_tensor = zeros(6, 6)
        MGES = m1 + m2
        for i in 1:size(M_tensor)[1]
            if i <= Int(size(M_tensor)[1]/2)
                M_tensor[i, i] = m1
            else
                M_tensor[i, i] = m2
            end
        end
        V_COM = get_VCOM(m1, m2, V1, V2)
        Ekin_COM = 0.5 * MGES*u"u" * dot(V_COM*u"a0_au*Eh_au/ħ_au",V_COM*u"a0_au*Eh_au/ħ_au")
        return uconvert(u"eV",(Ekin_COM))
    end

    function remove_surface(ase_atoms)
        for i in 1:length(ase_atoms)-2 # remove surface 
            ase_atoms.pop(0)
        end
        return ase_atoms
    end

    function velocity_transform(friction_atoms_id, atoms, velocities)
        modes = get_modes(atoms,friction_atoms_id)
        velocities_flat = vcat(velocities...)
        transformed_velocities = transpose(modes)*velocities_flat
    
        return transformed_velocities
    end

    """
    Output the end point of each trajectory.
    """
    struct OutputTrajectory 
        atoms
        path
        surface
        e_tran
        model_pes_calc
        model_eft
        step
        n_atoms_layer
        scat_cutoff
        selected_trajs
    end

    function (out::OutputTrajectory)(sol, i)
        e_eh_dd = 0*u"eV"
        e_eh_zz = 0*u"eV"
        traj_id_start = parse(Int64, ENV["TRAJ_START"])
        atoms_in_out = []
        len_sol = length(sol.u)
        traj_id = out.selected_trajs[i] # traj_id_start + i - 1

        path_out = "$(out.path)/plots_data_h2$(out.surface)_E$(out.e_tran)_traj$(traj_id).xyz"
        
        if isfile(path_out) == false

            for (n, e) in enumerate(sol.u)
                atoms_cur = deepcopy(out.atoms)
                atoms_cur = remove_surface(atoms_cur)
                atoms_full = deepcopy(out.atoms)
                posits = NQCDynamics.get_positions(e)
                vels = NQCDynamics.get_velocities(e)
                posits_h2 = deepcopy(posits[:,end-1:end])
                vels_h2 = deepcopy(vels[:,end-1:end])

                posits = transpose(posits)*u"a0_au"
                posits = uconvert.(u"Å",posits)
                posits = posits./u"Å"
                vels = transpose(vels)

                atoms_cur.set_positions(posits[end-1:end,:])
                atoms_cur.set_velocities(vels[end-1:end,:])
                atoms_full.set_positions(posits)
                atoms_full.set_velocities(vels)

                atoms, R, cell =  NQCBase.convert_from_ase_atoms(atoms_full)
                friction_ids = [length(atoms_full)-1,length(atoms_full)]
                F = zeros(3*length(atoms_full), 3*length(atoms_full))
                NQCModels.FrictionModels.friction!(out.model_eft, F, R)
                mass_h = atoms.masses[end] # H mass
                F = F[(friction_ids[1]-1)*3+1:friction_ids[2]*3,(friction_ids[1]-1)*3+1:friction_ids[2]*3]
                F .= auconvert.(u"ps^-1", F/(sqrt(mass_h)*sqrt(mass_h)))/u"ps^-1" # IMPORTANT STEP - CONVERT UNITS TO ps^-1!!!

                modes = get_modes(atoms_full, friction_ids)
                F_int = transpose(modes)*(F*modes) # *0.103642698490706 # (if you want to get meVpsA-2 for h2cu)
                F_dd = F_int[1,1] # in ps-1
                F_zz = F_int[6,6] # in ps-1
                # F_dz = F_int[1,6] # in ps-1
                vel_transf = velocity_transform(friction_ids, atoms_full, [vels[end-1,:],vels[end,:]])
                vel_dd_i = abs.(vel_transf[1])
                vel_zz_i = abs.(vel_transf[6])
                e_eh_dd_i = uconvert(u"eV",((vel_dd_i*u"a0_au*Eh_au/ħ_au")^2)*F_dd*u"u*ps^-1"*out.step*u"ps") # CHECK IF THE "u" should be here + check transformation matrix!!! 
                e_eh_zz_i = uconvert(u"eV",((vel_zz_i*u"a0_au*Eh_au/ħ_au")^2)*F_zz*u"u*ps^-1"*out.step*u"ps") # CHECK IF THE "u" should be here + check transformation matrix!!!
                e_eh_dd += e_eh_dd_i
                e_eh_zz += e_eh_zz_i
                # vel_sq_dd = uconvert(u"(Å^2)/(fs^2)",((vel_dd_i*u"a0_au*Eh_au/ħ_au")^2))
                # vel_sq_zz = uconvert(u"(Å^2)/(fs^2)",((vel_zz_i*u"a0_au*Eh_au/ħ_au")^2))

                if n == 1 || n == len_sol

                    # Set up sim for evaluating v and J states (EBK)
                    atoms_cur.set_calculator(out.model_pes_calc) # just H2
                    model_h2 = AdiabaticASEModel(atoms_cur)
                    xatoms, xpositions, xcell = NQCDynamics.convert_from_ase_atoms(atoms_cur)
                    sim = Simulation{Classical}(xatoms, model_h2, cell=xcell)
                    

                    # Get all energies
                    m0 = atoms_cur.get_masses()[1]
                    m1 = atoms_cur.get_masses()[2]
                    e_transl = get_proj_COM_Ekin(m0,m1,vels[end-1,:],vels[end,:])
                    e_pot = potential_energy(atoms_cur,out.model_pes_calc)
                    e_kin = kinetic_energy_ase(atoms_cur,out.model_pes_calc)
                    e_total = e_pot + e_kin
                    e_int = e_total - e_transl

                    if n == len_sol # last step
                        # check if scatter or DC
                        top_surface_avg_z = sum(NQCDynamics.get_positions(e)[3,end-Int(out.n_atoms_layer)-1:end-2])/out.n_atoms_layer
                        h2_surface_distance = minimum(NQCDynamics.get_positions(e)[3,end-1:end]) - top_surface_avg_z # min H2 z - max Cu z value
                        if h2_surface_distance >= out.scat_cutoff  # Scattering event
                            scatter_info = 1
                            vJs = QuantisedDiatomic.quantise_diatomic(sim, vels_h2, posits_h2)
                        else
                            scatter_info = 0
                            vJs = [-1,-1]
                        end
                        atoms_full.info = Dict("surface"=>out.surface,
                                            "e_kin"=>e_kin/oneunit(e_kin),
                                            "e_pot"=>e_pot/oneunit(e_pot),
                                            "e_transl"=>e_transl/oneunit(e_transl),
                                            "e_int"=>e_int/oneunit(e_int),
                                            "e_tot"=>e_total/oneunit(e_total),
                                            "traj_id"=>traj_id,
                                            "v_state"=>vJs[1],
                                            "j_state"=>vJs[2],
                                            "scatter"=>scatter_info,
                                            "e_eh_dd"=>e_eh_dd/oneunit(e_eh_dd),
                                            "e_eh_zz"=>e_eh_zz/oneunit(e_eh_zz)
                                            )
                    else # n == 1
                        vJs = QuantisedDiatomic.quantise_diatomic(sim, vels_h2, posits_h2)
                        atoms_full.info = Dict("surface"=>out.surface,
                                            "e_kin"=>e_kin/oneunit(e_kin),
                                            "e_pot"=>e_pot/oneunit(e_pot),
                                            "e_transl"=>e_transl/oneunit(e_transl),
                                            "e_int"=>e_int/oneunit(e_int),
                                            "e_tot"=>e_total/oneunit(e_total),
                                            "traj_id"=>traj_id,
                                            "v_state"=>vJs[1],
                                            "j_state"=>vJs[2]
                                            )
                    end
                    
                    push!(atoms_in_out, atoms_full)
                end
            end
            
            path_out = "$(out.path)/plots_data_h2$(out.surface)_E$(out.e_tran)_traj$(traj_id).xyz"
            io.write(path_out, atoms_in_out)
        end

        return (i,last(sol.u)) # return last trajectory
    end




    """
    ------------------------------------------------------------------------
    - Running Ensemble (x_cores) simulation by using the initial conditions
    ------------------------------------------------------------------------
    """

    ############################################
    ################# SETTINGS #################
    ############################################ 

    surface = ENV["SET_SURFACE"] # surface
    vjs = [parse(Int64, ENV["SET_QN_V"]), parse(Int64, ENV["SET_QN_J"])] # v and J states
    temp_surf = ENV["SET_TEMPERATURE"] # temperature of surface
    e_tran = parse(Float64, ENV["SET_E_TRANSL"]) # translational/collision energy
    #temp_el = 0u"K" # electronic temperature
    n_layers_metal = 6


    ############################################
    ############# INITIAL SETTINGS #############
    ############################################ 

    icond_f = ENV["H2CU_ML_INIT_COND"]
    icond_str = ENV["H2CU_ML_INPUT_STR"]
    output_f = ENV["H2CU_OUTPUTS"]

    mkpath(output_f)
    ml_model_path = ENV["PES_MODEL_PATH"] #  "$(ENV["H2CU_ML_MODELS"])/$(model_folder)"
    ml_model_ver = ENV["PES_MODEL_VER"]
    eft_model_path = ENV["EFT_MODEL_PATH"]

    # SAVING DATAPOINTS FOR ADAPTIVE LEARNING
    save_trajectory = true # choose if you want to save structures from entire trajectory
    v_models_error = 0.025 # minimum error (std) between potential energies of multiple models, of the structure that will be saved for adaptive sampling

    # SIMULATION DETAILS
    height = 7.0 # ang
    traj_start = parse(Int64, ENV["TRAJ_START"]) # starting trajectory (based on initial conditions)
    traj_end = parse(Int64, ENV["TRAJ_END"]) # last trajectory (based on initial conditions)
    traj_num = traj_end - traj_start + 1 # number of trajectories to run
    max_time_fs = 3000 # simulation time / fs
    max_time = max_time_fs*u"fs"
    step = 0.1*u"fs" # simulation step
    step_ps = (step*0.001)/u"fs"
    scat_cutoff = ang_to_au(7.2) # termination condition (h2 height)
    dist_cutoff = ang_to_au(2.25) # termination condition (h2 bond length)
    slab_outside_max_len = ang_to_au(10)

    ############################################
    ########### READING DISTRIBUTION ###########
    ############################################ 

    println("Reading distribution ...")
    file_name = "Et$(e_tran)_v$(vjs[1])_J$(vjs[2])/distr_Et$(e_tran)_v$(vjs[1])_J$(vjs[2])"
    file_name_str = "Et$(e_tran)_v$(vjs[1])_J$(vjs[2])/traj_Et$(e_tran)_v$(vjs[1])_J$(vjs[2])"
    system = load("$(icond_f)/$(file_name).jld2")
    ase_atoms = io.read("$(icond_f)/$(file_name_str).xyz@1")
    cell = system["cell"]
    distribution = system["dist"]
    atoms = system["atoms"]
    atoms = NQCDynamics.Atoms(atoms.types)
    atoms_all = []
    
    models_as = []
    errors = []
    cur_folder = "$(output_f)/$(ml_model_ver)"
    mkpath(cur_folder)
    n_atoms_layer = (length(ase_atoms)-2)/n_layers_metal # if molecule is 2 atoms
    ase_atoms.set_constraint(get_constraints_ase(ase_atoms,1:(floor(Int, n_atoms_layer*2))))
    ase_atoms_jl = ase_atoms.copy()
    ase_atoms_jl.pop(-1)
    ase_jl = ASE.ASEAtoms(ase_atoms_jl)
    atoms_julip = JuLIP.Atoms(ase_jl)

    ############################################
    ############## SET SIMULATION ##############
    ############################################ 

    # LOAD PES MODEL
    pes_model = mace_model(ml_model_path, ase_atoms)
    calculator_pes = mace_calc.MACECalculator(model_path=ml_model_path, device="cpu", default_dtype="float32")

    # LOAD EFT ODF MODEL
    friction_ids = [length(ase_atoms)-1,length(ase_atoms)]
    eft_model_ml = ace_model(eft_model_path, atoms_julip)
    density_model = AceLDFA(eft_model_ml; density_unit=u"Å^-3")
    eft_model = LDFAFriction(density_model, atoms; friction_atoms=friction_ids)

    # COMBINE PES + EFT MODELS FOR MDEF
    model = CompositeFrictionModel(pes_model,eft_model)
    sim = Simulation{MDEF}(atoms, model, cell=cell, temperature=parse(Float64,temp_surf)*u"K")


    # SET TRAJECTORY TERMINATOR FUNCTION 
    terminator = TrajectoryTerminator([length(ase_atoms)-1,length(ase_atoms)], scat_cutoff, dist_cutoff, slab_outside_max_len, 
                            [[ang_to_au(0.0), cell.vectors[1,1] + cell.vectors[1,2]], [ang_to_au(0.0), cell.vectors[2,2]], [ang_to_au(0.0), ang_to_au(15.0)]],Int(n_atoms_layer))

    terminate_cb = DynamicsUtils.TerminatingCallback(terminator)
    
end # @everywhere


############################################
############## RUN SIMULATION ##############
############################################ 

println("Initialize ...  ")
println("... Running simulation ... ")

selection_all = []
for t_id in traj_start:traj_end
    path_out = "$(cur_folder)/plots_data_h2$(surface)_E$(string(e_tran))_traj$(t_id).xyz"
    if isfile(path_out) == false
        append!(selection_all,t_id)
    end
end


@time ensemble = Ensembles.run_dynamics(sim, (0.0, max_time), distribution; selection=selection_all, dt=step, trajectories=length(selection_all), 
                        output=OutputTrajectory(ase_atoms, cur_folder, surface, string(e_tran), calculator_pes, eft_model, step_ps, n_atoms_layer, scat_cutoff, selection_all),
                        callback=terminate_cb, ensemble_algorithm=EnsembleDistributed(), saveat=(0.0:austrip(0.1*u"fs"):austrip(max_time_fs*u"fs")))

