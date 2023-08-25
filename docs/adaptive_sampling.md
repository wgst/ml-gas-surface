---
layout: default
title: Adaptive sampling
nav_order: 3
---

The adaptive sampling procedure for gas-surface dynamics is described by [Stark et al.](https://arxiv.org/abs/2305.10873). We will briefly describe the process here, starting with the scheme shown below.

<img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/images/adaptive_sampling_scheme.png?raw=true" width="500">

## Initial database

## Training MLIPs
In order to go to the next step of adaptive sampling, we need to train at least 3 MLIPs. Usually, this can be done by just training on different random training-validation-test splits.

## High-energy structure search (Julia code)
After training the models, we can finally start searching for high-error structures in our system. An excellent environment to run such simulations is [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl/) within [Julia](https://julialang.org).

In order to run the dynamics, we recommend creating the initial conditions (initial positions and velocities) separately. This part is very well explained in the [NQCDynamics.jl documentation](https://nqcd.github.io/NQCDynamics.jl/dev/).
Our initial conditions scripts for dissociative chemisorption could be also of help ([link1](https://github.com/wgst/ml-gas-surface/blob/main/scripts/dynamics/dissociative_chemisorption/initial_conditions/molecule%2Bsurface/initial_conditions_3x3_6lrs.jl), [link2](https://github.com/wgst/ml-gas-surface/blob/main/scripts/dynamics/dissociative_chemisorption/adaptive_sampling/high_error_structure_search/sticking_prob_save_str.jl)).

Having both the ML models and the initial conditions, we can finally run the high-error structure search. Below we will show an example script, which can also be accessed directly [here](https://github.com/wgst/ml-gas-surface/blob/main/scripts/dynamics/dissociative_chemisorption/adaptive_sampling/high_error_structure_search/sticking_prob_save_str.jl).

```jl
using NNInterfaces
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
Function to compute H-H distance atoms
"""
function dh2(p) 
    norm(p[:,end].-p[:,end-1])  
end

"""
Function for removing surface
"""
function remove_surface(ase_atoms)
    n_a = length(ase_atoms)
    for i in 1:n_a-2
        ase_atoms.pop(0)
    end
    return ase_atoms
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
Function for processing the results
"""
function ensemble_processing(ensemble, dist_cutoff, scat_cutoff, atoms, cell, model_h2, surface, e_tran, cur_folder, n_atoms_layer)
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
        h2_distance = dh2(NQCDynamics.get_positions(e))
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

"""
Output the end point of each trajectory.
"""
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

    return last(sol.u) # return last trajectory
end




"""
------------------------------------------------------------------------
- Running Ensemble (x_cores) simulation by using the initial conditions
------------------------------------------------------------------------
"""

############################################
############# INITIAL SETTINGS #############
############################################ 

es_tran = [0.200, 0.400, 0.500, 0.600, 0.750, 0.850] # v0
#es_tran = [0.200, 0.300, 0.400, 0.500, 0.600, 0.750] # v1

model_folder = ENV["MODEL_FOLDER"]
icond_f = "$(ENV["H2CU_ML_INIT_COND"])/output/$(model_folder)/"
icond_str = "$(ENV["H2CU_ML_INIT_COND"])/inputs/"
output_f = "$(ENV["H2CU_ML_DYN"])/output/$(model_folder)/"
mkpath(output_f)
ml_model_f = "$(ENV["H2CU_ML_MODELS"])/$(model_folder)"

# SAVING DATAPOINTS FOR ADAPTIVE LEARNING
save_errors = true # choose if you want to save structures with min error of v_models_error for adaptive sampling 
v_models_error = 0.025 # minimum error (std) of potential energy predictions made by multiple models, of the structure that will be saved for adaptive sampling

# SIMULATION DETAILS
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

############################################
################# SETTINGS #################
############################################ 

surface = "cu111" # surface
e_tran = es_tran[1] # translational/collision energy
vjs = [0,1] # v and J states
temp_surf = "925" # temperature of surface
n_layers_metal = 6

############################################
########### READING DISTRIBUTION ###########
############################################ 

println("Reading distribution ...")
file_name = "Et$(e_tran)_v$(vjs[1])_J$(vjs[2])/distr_Et$(e_tran)_v$(vjs[1])_J$(vjs[2])"
system = load("$(icond_f)$(surface)/T$(temp_surf)K/$(file_name).jld2")
ase_atoms = io.read("$(icond_str)$(surface)_h$(height)_full_$(temp_surf)K.in")
cell = system["cell"]
distribution = system["dist"]
atoms = system["atoms"]
atoms = NQCDynamics.Atoms(atoms.types)
atoms_all = []
models_mult = []
models_mult_paths = [] # this can be done in couple of ways, but e.g. add paths to different models here
errors = []
cur_folder = "$(output_f)$(surface)/v$(vjs[1])J$(vjs[2])/T$(temp_surf)/"
mkpath(cur_folder)
n_atoms_layer = (length(ase_atoms)-2)/n_layers_metal # if molecule is 2 atoms

############################################
############## SET SIMULATION ##############
############################################ 

# LOAD MODELS FOR ADAPTIVE LEARNING PROCEDURE
if save_errors == true
    for (i, m) in enumerate(models_mult_paths) # 
        append!(models_mult, [schnet_model_pes(models_mult_paths[i], ase_slab.copy())])
    end
end

# LOAD MODELS
model = schnet_model_pes(ml_model_f, ase_atoms)
sim = Simulation{Classical}(atoms, model, cell=cell)

# SET TRAJECTORY TERMINATOR FUNCTION 
terminator = TrajectoryTerminator([length(ase_atoms)-1,length(ase_atoms)], scat_cutoff, dist_cutoff, slab_outside_max_len, 
                        [[ang_to_au(0.0), cell.vectors[1,1] + cell.vectors[1,2]], [ang_to_au(0.0), cell.vectors[2,2]], [ang_to_au(0.0), ang_to_au(15.0)]],Int(n_atoms_layer))

terminate_cb = DynamicsUtils.TerminatingCallback(terminator)

############################################
############## RUN SIMULATION ##############
############################################ 

println("Initialize ...  ")
println("... Running simulation ... ")
@time ensemble = Ensembles.run_dynamics(sim, (0.0, max_time), distribution; selection=traj_start:traj_end, dt=step, trajectories=traj_num, 
                        output=OutputTrajectory(atoms, cell, cur_folder, surface, string(e_tran), models_mult, traj_start, v_models_error, save_errors),
                        callback=terminate_cb, ensemble_algorithm=EnsembleDistributed(), saveat=(0.0:austrip(1.0*u"fs"):austrip(max_time_fs*u"fs")))


############################################
############## POSTPROCESSING ##############
############################################ 

# Collect final output results
#------------------------------
# create new sim just for H2 (for quantise_diatomic)
ase_atoms = remove_surface(ase_atoms)
model_h2 = schnet_model_pes(ml_model_f, ase_atoms)
output_data, atoms_all, n_scat, n_reac, n_nondef = ensemble_processing(ensemble, dist_cutoff, scat_cutoff, atoms, cell, model_h2, surface, e_tran, cur_folder, Int(n_atoms_layer))

# Calculating dissociation probability:
n_traj_sr = n_scat + n_reac
n_traj_all = n_scat + n_reac + n_nondef
prob_reac = n_reac/n_traj_sr
prob_reac_all = n_reac/n_traj_all

println("... Ending simulation and printing external files ...")
# Print dissociation probability
labels = ["n_scattering: ", "n_reaction: ", "n_nondefined: ", "reaction_probability: ", "reaction_probability_all: "]
results =[n_scat, n_reac, n_nondef, prob_reac, prob_reac_all]
writedlm("$(cur_folder)$(surface)_Et$(e_tran)_results_$(traj_start)to$(traj_end).log", zip(labels,results))
```


## Clustering (Python code)
The database of high-error structures usually contains too many data points. Fortunately, clustering methods, such as k-means, can help to find the most informative data points and add only them to our database. To do that, we prepared a Python notebook, which can be found [in this repository](https://github.com/wgst/ml-gas-surface/blob/main/scripts/dynamics/dissociative_chemisorption/adaptive_sampling/high_error_structure_clusters/choosing_datapoints_h2cu111_v0_clean.ipynb). Its contents with an explanation are also included below.

### Import database with high-error structures

```py
import matplotlib
import matplotlib.pyplot as plt
from ase.io import write, read
import numpy as np

db_p = 'output_h2cu111_example.db' # Input db path
db_out_p = 'h2cu_clust_centr.xyz' # Path for saving cluster centres
ncluster = 80 # Number of clusters (final structures that will be added to our database)
# Read all high error structures
struct_all = read(f'{db_p}@:')
```

### Define descriptors (inverse distances)

```py
dist_all = []
H2_dist = np.zeros((len(struct_all),3))
# For H2 get the minimum distance to all atoms
for j in range(len(struct_all)):
    dist_cur = struct_all[j].get_all_distances(mic=True)
    # Save dist_all list for inverse distance matrix later
    dist_all.append(dist_cur)
    # Minimal descriptor
    H2_dist[j][0] = dist_cur[-1,-2] # Distance between H-H
    H2_dist[j][1] = np.min(dist_cur[-1,:-2]) # Min distance between H1-Cu
    H2_dist[j][2] = np.min(dist_cur[-2,:-2]) # Min distance between H2-Cu

natoms = len(struct_all[0].get_all_distances())
invd = np.zeros((len(dist_all),natoms,natoms))

# Make 1s out of the diagonal
for i in range(len(dist_all)):
    np.fill_diagonal(dist_all[i],1.0)
    invd[i] = np.ones((natoms,natoms))/dist_all[i]

# We will use invd[:,-2:], which refers to the inverse distances between both H atoms and all atoms
invd_reduced = invd[:,-2:]
invd = invd.reshape(len(dist_all),natoms*natoms)
invd_reduced = invd_reduced.reshape(len(dist_all),-1)
descr = invd_reduced
```

### Use PCA for dimension reduction (optional)

```py
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(descr)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
X_new = pca.transform(descr)

plt.xlabel("PCA components")
plt.ylabel("Variance ratio")

plt.bar(np.arange(10),pca.explained_variance_ratio_, edgecolor='black')
```

<img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/images/h2cu_pca_analysis.png?raw=true" width="300">

```py
descr = X_new[:,:2]
fig, ax = plt.subplots(1, 1)

ax.scatter(descr[:,0], descr[:,1], edgecolors='white', linewidths=0.4, label="datapoints", alpha=0.8)

plt.xlabel("PC1")
plt.ylabel("PC2")
fig.set_figheight(6.0)
fig.set_figwidth(6.0)

plt.show()
```

<img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/images/plot_PCA_h2cu111_E_all_v0.png?raw=true" width="500">

```py
from sklearn.cluster import MiniBatchKMeans

niter = 1000000
minibatch = 1800 # Batch size (how many data points are treated in the algorithm at once, reduce if you run out of memory)
kmeans = MiniBatchKMeans(n_clusters=ncluster,
                        init="k-means++",
                        max_iter=niter,
                        batch_size=minibatch)
# Fit the clustering model
km = kmeans.fit(np.array(descr).reshape(-1,1))
# Use the clustering model to predict the cluster number of each data point
indices = km.fit_predict(descr)
```

### Finding centers of clusters

```py
from sklearn.metrics.pairwise import pairwise_distances_argmin

centroid = kmeans.cluster_centers_
b = np.inf
ind = pairwise_distances_argmin(centroid, descr)
print(ind)

geometries = []
X_centers = []
H2_centers = []

for i in range(len(ind)):
    geometries.append(struct_all[ind[i]])
    X_centers.append(X_new[ind[i]])
    H2_centers.append(H2_dist[ind[i]])

# Save the chosen center structures
write(db_out_p, geometries)

H2_centers = np.array(H2_centers)
X_centers = np.array(X_centers)
```

### Plotting the final centers within PCs

```py
fig, ax = plt.subplots(1, 1)

ax.scatter(X_new[:,0], X_new[:,1],c=indices, edgecolors='white', linewidths=0.4, label="datapoints", alpha=0.8)
ax.scatter(X_centers[:,0],X_centers[:,1],color="white",label="centers",edgecolors='black',linewidths=0.7,marker="X", alpha=0.8)

plt.legend(fancybox=True,framealpha=1,edgecolor='black',handletextpad=0.05,borderpad=0.3,handlelength=1.2,columnspacing=0.4,labelspacing=0.2,ncol=1,loc=1, fontsize="medium") #, bbox_to_anchor=(1.75, 1.02))
plt.xlabel("PC1")
plt.ylabel("PC2")
fig.set_figheight(6.0)
fig.set_figwidth(6.0)

plt.show()
```
<img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/images/plot_PC1PC2_h2cu111_E_all_v0.png?raw=true" width="500">

### Plotting centers of clusters within H-H and H1-Cu distances

```py

fig, ax = plt.subplots(1, 1)

ax.scatter(H2_dist[:,0], H2_dist[:,1],c=indices,edgecolors='white', linewidths=0.4, label="datapoints", alpha=0.8)
ax.scatter(H2_centers[:,0],H2_centers[:,1],color="white",label="centers",edgecolors='black',linewidths=0.7,marker="X", alpha=0.8)

plt.legend(fancybox=True,framealpha=1,edgecolor='black',handletextpad=0.05,borderpad=0.3,handlelength=1.2,columnspacing=0.4,labelspacing=0.2,ncol=1,loc=1, fontsize="medium") #, bbox_to_anchor=(1.75, 1.02))
plt.xlabel("H1-H2 distance / Å")
plt.ylabel("H1-Cu distance / Å")
fig.set_figheight(6.0)
fig.set_figwidth(6.0)

plt.show()

```

<img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/images/plot_PC_distances_h2cu111_E_all_v0.png?raw=true" width="500">





