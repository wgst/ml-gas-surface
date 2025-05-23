using Pkg
Pkg.activate("/home/chem/msrmnk/ace_friction_2_new")
Pkg.instantiate()

using LinearAlgebra
using ACEds.FrictionModels

using ACE: scaling, params
using ACEds
using ACEds.FrictionFit
using ACEds.DataUtils
using ACEds.Utils: reinterpret
using Flux
using Flux.MLUtils
using ACE
using ACEds: ac_matrixmodel
using Random
using ACEds.Analytics
using ACEds.FrictionFit
using ACEds.FrictionFit: weighted_l2_loss
using ACEds.Analytics: error_stats, plot_error, plot_error_all, friction_entries
using ACEds.MatrixModels
using PyPlot
using JuLIP
using CUDA
using JLD2

cuda = CUDA.functional()

##########################################
########### IMPORT THE DATABASE ##########
##########################################

println("Reading the inputs...")

output_path = "$(ENV["OUTPUT_PATH"])"
mkpath("$(output_path)1_plots/")

##########################################
# Partition data into train and test set #
##########################################

split_no = ENV["SPLIT_NO"]
filename = ENV["DB_EFT"]
rdata = ACEds.DataUtils.json2internal(filename)
shuffle!(rdata)
n_train = parse(Int64, ENV["N_TRAIN"])
n_test = parse(Int64, ENV["N_TEST"])
data = Dict("train" => rdata[1:n_train], "test"=> rdata[n_train+1:n_train+n_test]);

test_data=[at[:at] for at in rdata[n_train+1:n_train+n_test]]
write_extxyz("$(output_path)h2cu_test.xyz", test_data)

##########################################
########### DEFS AND SETTINGS ############
##########################################

species_friction = [:H]
species_mol = [:H]
species_env = [:Cu]
rcut = parse(Float64, ENV["CUTOFF"])
coupling= RowCoupling() # TEST COLUMN # RowCoupling() # or ColumnCoupling
# m_inv = ac_matrixmodel(ACE.Invariant(),species_friction,species_env, coupling; n_rep = 2, rcut_on = rcut, rcut_off = rcut, maxorder_on=2, maxdeg_on=5,

# TRAINING
nepochs = parse(Int64, ENV["N_EPOCHS"])

weight_diag = parse(Float64, ENV["WEIGHT_DIAG"])
weight_subdiag = parse(Float64, ENV["WEIGHT_SUBDIAG"])
weight_offdiag = parse(Float64, ENV["WEIGHT_OFFDIAG"])

bond_weight = parse(Float64, ENV["MATRIX_BOND_WEIGHT"]) # default: 0.5
n_rep_inv = parse(Int64, ENV["N_REP_INV"]) # default: 2
n_rep_cov = parse(Int64, ENV["N_REP_COV"]) # default: 3
n_rep_equ = parse(Int64, ENV["N_REP_EQU"]) # default: 2
max_order_on = parse(Int64, ENV["MAX_ORDER"]) # default: 2
max_degree_on = parse(Int64, ENV["MAX_DEGREE"]) # default: 5

species_maxorder_dict_on_inv = Dict(:H => 1)
species_weight_cat_on_inv = Dict(:H => .75, :Cu=> 1.0)
species_maxorder_dict_off_inv = Dict(:H => 0)
species_weight_cat_off_inv = Dict(:H => 1.0, :Cu=> 1.0)

species_maxorder_dict_on_cov = Dict(:H => 1)
species_weight_cat_on_cov = Dict(:H => .75, :Cu=> 1.0)
species_maxorder_dict_off_cov = Dict(:H => 0)
species_weight_cat_off_cov = Dict(:H => 1.0, :Cu=> 1.0)

species_maxorder_dict_on_equ = Dict(:H => 1)
species_weight_cat_on_equ = Dict(:H => .75, :Cu=> 1.0)
species_maxorder_dict_off_equ = Dict(:H => 0)
species_weight_cat_off_equ = Dict(:H => 1.0, :Cu=> 1.0)

println("Chosen settings:")
println("Model no: $(split_no)")
# println("Number of training datapoints: $(n_train)")
# println("Number of test datapoints: $(length(rdata)-n_train)")
println("Cutoff distance: $(rcut)")
println("Number of epochs: $(nepochs)")
println("Weight for diagonal elements: $(weight_diag)")
println("Weight for subdiagonal elements: $(weight_subdiag)")
println("Weight for offdiagonal elements: $(weight_offdiag)")
println("Bond weight for matrices: $(bond_weight)")
println("Number of repetitions (invariant matrix): $(n_rep_inv)")
println("Number of repetitions (covariant matrix): $(n_rep_cov)")
println("Number of repetitions (equivariant matrix): $(n_rep_equ)")
println("Max body order: $(max_order_on)")
println("Max polynomial degree: $(max_degree_on)")

######################################################
# DEFINE INVARIANT, COVARIANT AND EQUIVARIANT MATRIX #
######################################################

m_inv = ac_matrixmodel(ACE.Invariant(),species_friction,species_env,coupling,species_mol; n_rep = n_rep_inv, rcut_on = rcut, rcut_off = rcut, maxorder_on=max_order_on, maxdeg_on=max_degree_on,
        species_maxorder_dict_on = species_maxorder_dict_on_inv, 
        species_weight_cat_on = species_weight_cat_on_inv,
        species_maxorder_dict_off = species_maxorder_dict_off_inv, 
        species_weight_cat_off = species_weight_cat_off_inv,
        bond_weight = bond_weight
    );
# m_cov = ac_matrixmodel(ACE.EuclideanVector(Float64),species_friction,species_env,coupling,species_mol;n_rep=n_rep_cov, rcut_on = rcut, rcut_off = rcut, maxorder_on=max_order_on, maxdeg_on=max_degree_on,
#         species_maxorder_dict_on = species_maxorder_dict_on_cov, 
#         species_weight_cat_on = species_weight_cat_on_cov,
#         species_maxorder_dict_off = species_maxorder_dict_off_cov, 
#         species_weight_cat_off = species_weight_cat_off_cov,
#         bond_weight = bond_weight
#     );
m_equ = ac_matrixmodel(ACE.EuclideanMatrix(Float64),species_friction,species_env,coupling,species_mol;n_rep=n_rep_equ, rcut_on = rcut, rcut_off = rcut, maxorder_on=max_order_on, maxdeg_on=max_degree_on,
        species_maxorder_dict_on = species_maxorder_dict_on_equ, 
        species_weight_cat_on = species_weight_cat_on_equ,
        species_maxorder_dict_off = species_maxorder_dict_off_equ, 
        species_weight_cat_off = species_weight_cat_off_equ,
        bond_weight = bond_weight
    );


##########################################
############ INITIALISE MODEL ############
##########################################
println("Initialising model...")
# FrictionModel
eft_model = FrictionModel((m_inv,m_equ))
model_ids = get_ids(eft_model)

# Create friction data in internally used format
fdata =  Dict(
    tt => [FrictionData(d.at,
            d.friction_tensor, 
            d.friction_indices; 
            weights=Dict("diag" => weight_diag, "sub_diag" => weight_subdiag, "off_diag"=>weight_offdiag)) for d in data[tt]] for tt in ["test","train"]
);

c = params(eft_model;format=:matrix, joinsites=true)
flux_eft_model = FluxFrictionModel(c)


# Create preprocessed data including basis evaluations that can be used to fit the model
flux_data = Dict(
    tt => flux_assemble(fdata[tt], 
                        eft_model, 
                        flux_eft_model; 
                        weighted=true, 
                        matrix_format=:dense_scalar) for tt in ["train","test"]
) # crushed here (probably memory issue)

set_params!(flux_eft_model; sigma=1E-8)

#if CUDA is available, convert relevant arrays to cuarrays
if cuda
    flux_eft_model = fmap(cu, flux_eft_model)
end

loss_traj = Dict("train"=>Float64[], "test" => Float64[])
n_train, n_test = length(flux_data["train"]), length(flux_data["test"])

bsize = 10

optimizer = Flux.setup(Adam(1E-4, (0.99, 0.999)),flux_eft_model) # 1E-4 is a learning rate
train = [(friction_tensor=d.friction_tensor,B=d.B,Tfm=d.Tfm, W=d.W) for d in flux_data["train"]]
data_loader = cuda ? DataLoader(train |> gpu, batchsize=bsize, shuffle=true) : DataLoader(train, batchsize=bsize, shuffle=true)

mloss = weighted_l2_loss
Flux.gradient(mloss,flux_eft_model, train[1:2])[1]

##############################
###### TRAIN THE MODEL #######
##############################
println("Training model...")
epoch = 0
min_loss = Inf
epoch_min = 1
c_fit = params(flux_eft_model)
@time for _ in 1:nepochs
    global epoch, data_loader, flux_eft_model, optimizer, flux_data, min_loss, epoch_min, c_fit
    epoch+=1
    @time for d in data_loader
        ∂L∂m = Flux.gradient(mloss,flux_eft_model, d)[1]
        Flux.update!(optimizer,flux_eft_model, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], mloss(flux_eft_model,flux_data[tt]))
    end

    if loss_traj["train"][end] < min_loss
        min_loss = loss_traj["train"][end]
        c_fit = params(flux_eft_model)
        epoch_min = epoch
    end

    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
println("Epoch: $epoch, Abs Training Loss: $(loss_traj["train"][end]), Test Loss: $(loss_traj["test"][end])")
println("Epoch: $epoch, Avg Training Loss: $(loss_traj["train"][end]/n_train), Test Loss: $(loss_traj["test"][end]/n_test)")


##############################
###### PROCESS THE MODEL #####
##############################
println("Processing data...")

# The following code can be used to fit the model using the BFGS algorithm
# include("./additional-bfgs-iterations.jl")

set_params!(eft_model, c_fit)
# SAVE THE MODEL
save_dict("$(output_path)eft_ac.model", write_dict(eft_model))

# Evaluate different error statistics 


friction = friction_entries(data["train"], eft_model;  atoms_sym=:at)

df_abs, df_rel, df_matrix, merrors =  error_stats(data, eft_model; atoms_sym=:at, reg_epsilon = 0.01)

##############################
###### PLOT THE RESULTS ######
##############################
println("Plotting results...")

# # Plot errors model vs reference for diagonal, sub-diagonal and off-diagonal elements separately
# fig1, ax1 = plot_error(data, eft_model;merrors=merrors)
# fig1.savefig("$(output_path)1_plots/scatter-detailed-equ-cov.pdf", bbox_inches="tight")
# fig1.savefig("$(output_path)1_plots/scatter-detailed-equ-cov.png", bbox_inches="tight")

# # Plot overall errors model vs reference
# fig2, ax2 = plot_error_all(data, eft_model; merrors=merrors)
# fig2.savefig("$(output_path)1_plots/scatter-equ-cov.pdf", bbox_inches="tight")
# fig2.savefig("$(output_path)1_plots/scatter-equ-cov.png", bbox_inches="tight")

# Plot train and test epochs vs error "learning" curve
N_train, N_test = length(flux_data["train"]),length(flux_data["test"])
fig, ax = PyPlot.subplots()
ax.plot(loss_traj["train"]/N_train, label="train")
ax.plot(loss_traj["test"]/N_test, label="test")
ax.set_yscale(:log)
ax.legend()
display(fig)
fig.savefig("$(output_path)1_plots/train_vs_test.pdf", bbox_inches="tight")
fig.savefig("$(output_path)1_plots/train_vs_test.png", bbox_inches="tight")


# SAVE THE FRICTION
eft_test = friction_entries(data["test"], eft_model;  atoms_sym=:at)
JLD2.save("$(output_path)eft_vals_test.jld2", Dict("all"=>eft_test))

eft_train = friction_entries(data["train"], eft_model;  atoms_sym=:at)
JLD2.save("$(output_path)eft_vals_train.jld2", Dict("all"=>eft_train))

println("Done!")
