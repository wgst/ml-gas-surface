---
layout: default
title: High error structure search
parent: Adaptive sampling
nav_order: 8
---

# High-energy structure search
{: .no_toc }
The following section will include Julia-based code.

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

# Introduction
After training the models, we can finally start searching for high-error structures in our system. This is done by running molecular dynamics (MD) trajectories using forces from one of the trained models searching for potential energy surface holes across the trajectories. An excellent environment to run such simulations is [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl/) within [Julia](https://julialang.org).

In order to run the dynamics, we usually create the initial conditions (initial positions and velocities) separately, in order to be able to access it multiple times, if needed. This part is very well explained in the [NQCDynamics.jl documentation](https://nqcd.github.io/NQCDynamics.jl/dev/).
Our initial conditions scripts for dissociative chemisorption included in this repository could be also helpful ([link1](https://github.com/wgst/ml-gas-surface/blob/main/scripts/dynamics/dissociative_chemisorption/initial_conditions/molecule%2Bsurface/initial_conditions_3x3_6lrs.jl), [link2](https://github.com/wgst/ml-gas-surface/blob/main/scripts/dynamics/dissociative_chemisorption/adaptive_sampling/high_error_structure_search/sticking_prob_save_str.jl)).

Having both the ML-based PES models and a set of initial conditions, we can finally run MD with high-error structure search. Below we will show an example script, which can also be accessed directly [here](https://github.com/wgst/ml-gas-surface/blob/main/scripts/dynamics/dissociative_chemisorption/adaptive_sampling/high_error_structure_search/sticking_prob_save_str.jl).

## Importing packages
First, we import Julia-based and Python-based (PyCall) packages.
```jl
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

# Importing Python modules with PyCall
io = pyimport("ase.io")
db = pyimport("ase.db")
spk_utils = pyimport("schnetpack.utils")
spk_interfaces = pyimport("schnetpack.interfaces")
```

## Initial functions/structs
Next, we define a couple of functions and structs that will allow us to create certain MD callbacks or to analyse our results.

### Trajectory terminator
Below, we define an MD trajectory terminator, which stops the trajectory when certain conditions are met. Here, the trajectory is stopped if H<sub>2</sub> is above 'scat_cutoff' value or if the distance between hydrogens in H<sub>2</sub> is exceeds 'dist_cutoff' value.

```jl
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
```

### High-error structure saver
Struct *OutputTrajectory* with a function that is executed at the end of every trajectory. It allows calculating energies with all the models from *models_mult* list. After that, error (standard deviation) is calculated and compared to the manually chosen *v_models_error*, which is the minimum error required for the structure to be saved.

If the calculated error is higher than *v_models_error*, the structure is saved to the *db_out* database stored in a specified path.

{: .note }
In order to save memory, only the last structure is returned by our output function.

```jl
struct OutputTrajectory 
    atoms
    cell
    path
    surface
    e_tran
    models_mult
    selection_start
    v_models_error
    save_errors
end

function (out::OutputTrajectory)(sol, i)
    if out.save_errors == true
        atoms_all = []
        errors = []
        Vs = []
        cur_error = 0
        db_out = db.connect("$(out.path)output_h2$(out.surface)_E$(out.e_tran).db")

        for (n, e) in enumerate(sol.u)
            for (i1, m) in enumerate(out.models_mult)
                append!(Vs, potential(out.models_mult[i1], NQCDynamics.get_positions(e)))
            end
            cur_error = std(Vs)
            if cur_error >= out.v_models_error
                append!(errors, cur_error)
                cur_ase_atoms = NQCDynamics.convert_to_ase_atoms(out.atoms, NQCDynamics.get_positions(e), out.cell)
                append!(atoms_all, [cur_ase_atoms])
                data=Dict()
                data["surface"] = out.surface
                data["traj_id"] = out.selection_start + i - 1
                data["atoms_id"] = n
                data["error"] = cur_error
                data["e_transl"] = out.e_tran
                db_out.write(cur_ase_atoms, data=data)
            end
        end
    end

    return last(sol.u) # return last structure
end
```

### Postprocessing
Function *ensemble_processing* is used after MD simulation is over, to postprocess our simulation data. In this case of simulating dissociative chemisorption of H<sub>2</sub> on Cu surface, we iterate over all the trajectories (last structure of every trajectory), to check whether the trajectory ended up with a scattering of H<sub>2</sub> molecule from the surface or with the reaction (sticking), to be able to calculate sticking probability in further steps.

```jl
function ensemble_processing(ensemble, dist_cutoff, scat_cutoff, atoms, cell, surface, e_tran, cur_folder, n_atoms_layer)
    atoms_all = []
    output_data = []
    n_scat = 0
    n_reac = 0
    n_nondef = 0
    h2_distance = 0
    h2_surface_distance = 0
    scattering = 0
    reaction = 0
    non_defined = 0

    db_out = db.connect("$(cur_folder)output_last_str_h2$(surface)_E$(e_tran).db")

    for (n, e) in enumerate(ensemble) # for each traj
        e = e[:OutputTrajectory]
        scattering = 0
        reaction = 0
        non_defined = 0
        h2_distance = norm(NQCDynamics.get_positions(e)[:,end].-NQCDynamics.get_positions(e)[:,end-1])
        top_surface_avg_z = sum(NQCDynamics.get_positions(e)[3,end-Int(n_atoms_layer)-1:end-2])/n_atoms_layer
        h2_surface_distance = minimum(NQCDynamics.get_positions(e)[3,end-1:end]) - top_surface_avg_z # min H2 z - max Cu z value
        cur_ase_atoms = NQCDynamics.convert_to_ase_atoms(atoms, NQCDynamics.get_positions(e), cell)

        if h2_surface_distance >= scat_cutoff  # Scattering event
            n_scat += 1
            scattering = 1
        elseif h2_distance >= dist_cutoff            # DC event
            n_reac += 1
            reaction = 1
        else                              # Non-defined category
            n_nondef += 1
            non_defined = 1
        end

        data=Dict()
        data["surface"] = surface
        data["scattering"] = scattering
        data["reaction"] = reaction
        data["non_defined"] = non_defined
        db_out.write(cur_ase_atoms, data=data)
    end
    return output_data, atoms_all, n_scat, n_reac, n_nondef
end
```


### Model initializer
Function *schnet_model_pes* is used for creating an NQCDynamics model object utilizing ASE calculator (here SchNet calculator). 

{: .note }
This function can be replaced with any MLIP that can be accessed through ASE calculator.

```jl
function schnet_model_pes(model_path, cur_atoms)
    spk_model = spk_utils.load_model(model_path;map_location="cpu")
    model_args = spk_utils.read_from_json("$(model_path)/args.json")
    environment_provider = spk_utils.script_utils.settings.get_environment_provider(model_args,device="cpu")
    calculator = spk_interfaces.SpkCalculator(spk_model, energy="energy", forces="forces", environment_provider=environment_provider)
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end
```


## Simulation settings

Now, we set all the required paths and we create the output folders.

```jl
ml_model_f = "path/to/mlip/model"
icond_f = "path/to/initial/conditions/folder"
icond_str = "path/to/initial/structures/folder/"
output_f = "path/to/output/folder/"
mkpath(output_f)
```

We then choose settings for high-error structures saver (adaptive sampling).

```jl
save_errors = true # choose if you want to save structures with min error of v_models_error for adaptive sampling 
v_models_error = 0.025 # minimum error (std) of potential energy predictions made by multiple models, of the structure that will be saved for adaptive sampling
models_mult = []
models_mult_paths = [] # this can be done in couple of ways, but e.g. add paths to different models here
atoms_all = []
```

Next, we set all the simulation details.

```jl
height = 7.0 # H height / Ang 
traj_start = parse(Int64, ENV["TRAJ_START"]) # starting trajectory (based on initial conditions)
traj_end = parse(Int64, ENV["TRAJ_END"]) # last trajectory (based on initial conditions)
traj_num = traj_end - traj_start + 1 # number of trajectories to run
max_time_fs = 3000 # simulation time / fs
max_time = max_time_fs*u"fs"
step = 0.1u"fs" # simulation step
scat_cutoff = ang_to_au(7.1) # termination condition (h2 height)
dist_cutoff = ang_to_au(2.25) # termination condition (h2 bond length)
slab_outside_max_len = ang_to_au(10)
surface = "cu111" # surface
e_tran = 0.5 # translational/collision energy in eV
vjs = [0,1] # v and J states
temp_surf = "925" # temperature of surface
n_layers_metal = 6

cur_folder = "$(output_f)$(surface)/v$(vjs[1])J$(vjs[2])/T$(temp_surf)/"
mkpath(cur_folder)
```

## Reading the initial conditions files

```jl
file_name = "Et$(e_tran)_v$(vjs[1])_J$(vjs[2])/distr_Et$(e_tran)_v$(vjs[1])_J$(vjs[2])"
system = load("$(icond_f)$(surface)/T$(temp_surf)K/$(file_name).jld2")
ase_atoms = io.read("$(icond_str)$(surface)_h$(height)_full_$(temp_surf)K.in")
cell = system["cell"]
distribution = system["dist"]
atoms = system["atoms"]
atoms = NQCDynamics.Atoms(atoms.types)
n_atoms_layer = (length(ase_atoms)-2)/n_layers_metal # number of atoms in a metal layer (if molecule contains 2 atoms)
```

## Preparing the simulation
### Loading models
We first load models for high error structure search (adaptive sampling). To do that, we load all of the models with path included in the *models_mult_paths* list, using previously mentioned *schnet_model_pes* function.

```jl
if save_errors == true
    for (i, m) in enumerate(models_mult_paths) # 
        append!(models_mult, [schnet_model_pes(models_mult_paths[i], ase_slab.copy())])
    end
end
```

Then, we load a model used for MD and we initialize the *Simulation*.

```jl
model = schnet_model_pes(ml_model_f, ase_atoms)
sim = Simulation{Classical}(atoms, model, cell=cell)
```

### Set trajectory terminator
In order to terminate the trajectories whenever the simulation has reached its goal, we set the terminating callback, which will be executed at every step of the simulation.

```jl
terminator = TrajectoryTerminator([length(ase_atoms)-1,length(ase_atoms)], scat_cutoff, dist_cutoff, slab_outside_max_len, 
                        [[ang_to_au(0.0), cell.vectors[1,1] + cell.vectors[1,2]], [ang_to_au(0.0), cell.vectors[2,2]], [ang_to_au(0.0), ang_to_au(15.0)]],Int(n_atoms_layer))

terminate_cb = DynamicsUtils.TerminatingCallback(terminator)
```

## Running ensemble MD simulation
Finally, we use all the model parameters established in the previous steps and we run the ensemble dynamics using *Ensembles.run_dynamics* function.

{: .note }
All the trajectories will finish either after reaching the maximum simulation time *max_time_fs* or whenever callback function *terminate_cb* is satisfied.

```jl
@time ensemble = Ensembles.run_dynamics(sim, (0.0, max_time), distribution; selection=traj_start:traj_end, dt=step, trajectories=traj_num, 
                        output=OutputTrajectory(atoms, cell, cur_folder, surface, string(e_tran), models_mult, traj_start, v_models_error, save_errors),
                        callback=terminate_cb, ensemble_algorithm=EnsembleDistributed(), saveat=(0.0:austrip(1.0*u"fs"):austrip(max_time_fs*u"fs")))
```

## Postprocess
After the simulation is over, we execute the *ensemble_processing* function that allows us to calculate our final reaction (sticking) probability *prob_reac_all*.

```jl
output_data, atoms_all, n_scat, n_reac, n_nondef = ensemble_processing(ensemble, dist_cutoff, scat_cutoff, atoms, cell, surface, e_tran, cur_folder, Int(n_atoms_layer))

n_traj_all = n_scat + n_reac + n_nondef
prob_reac_all = n_reac/n_traj_all
```

We end the simulation by printing the output file containing our final results.

```jl
labels = ["n_scattering: ", "n_reaction: ", "n_nondefined: ", "reaction_probability: "]
results =[n_scat, n_reac, n_nondef, prob_reac_all]
writedlm("$(cur_folder)$(surface)_Et$(e_tran)_results_$(traj_start)to$(traj_end).log", zip(labels,results))
```

