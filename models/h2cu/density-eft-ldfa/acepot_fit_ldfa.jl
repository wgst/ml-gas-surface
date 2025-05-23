using Pkg
Pkg.activate("/home/chem/msrmnk/ace_pot2/cur")
Pkg.instantiate()

using ACEpotentials
using Random
using JLD2, NPZ

data_file = ENV["ACEpot_DATABASE"]
ver = ENV["ACEpot_MODEL_FOLDER"]

# SETTINGS
cur_name = "h2cu_ace"
# data_ids = npzread(ENV["ACEpot_DATA_IDS"])
N=parse(Int64, ENV["ACEpot_CORRELATION_ORDER"]) # 5
D=parse(Int64, ENV["ACEpot_POLYNOMIAL_DEGREE"]) # 10
r_cut=parse(Float64, ENV["ACEpot_CUTOFF"]) # 6.0
e_wght=parse(Float64, ENV["ACEpot_E_WEIGHT"])
f_wght=0

split_no=ENV["ACEpot_VER"]

if N==2
    if D==1
        TOT_DGRE = [6,2]
    elseif D==2
        TOT_DGRE = [4,4]
    elseif D==3
        TOT_DGRE = [10,6]
    elseif D==4
        TOT_DGRE = [14,8]
    elseif D==5
        TOT_DGRE = [16,12]
    end
elseif N==3
    if D==1
        TOT_DGRE = [6,4,2]
    elseif D==2
        TOT_DGRE = [4,4,2]
    elseif D==3
        TOT_DGRE = [12,10,8]
    elseif D==4
        TOT_DGRE = [14,10,6]
    elseif D==5
        TOT_DGRE = [18,14,10]
    end
elseif N==4
    if D==1
        TOT_DGRE = [10, 8, 6, 4]
    elseif D==2
        TOT_DGRE = [14, 12, 8, 6]
    elseif D==3
        TOT_DGRE = [16, 12, 10, 8]
    elseif D==4
        TOT_DGRE = [18, 12, 10, 8]
    elseif D==5
        TOT_DGRE = [18, 16, 8, 6]
    elseif D==6
        TOT_DGRE = [14, 10, 8, 6]
    elseif D==7
        TOT_DGRE = [14, 10, 6, 4]
    elseif D==8
        TOT_DGRE = [14, 8, 6, 4]
    end
elseif N==5
    if D==1
        TOT_DGRE = [10, 8, 6, 4, 2]
    elseif D==2
        TOT_DGRE = [14, 10, 8, 6, 4]
    elseif D==3
        TOT_DGRE = [16, 14, 10, 8, 6]
    elseif D==4
        TOT_DGRE = [18, 12, 8, 6, 4]
    elseif D==5
        TOT_DGRE = [14, 8, 8, 6, 4]
    elseif D==6
        TOT_DGRE = [14, 8, 6, 6, 6]
    elseif D==7
        TOT_DGRE = [14, 6, 6, 6, 6]
    end
end

println("NOW PROCEEDING WITH THE MODEL: $(ver)")
println("CORRELATION ORDER $(length(TOT_DGRE))")
println("POLYNOMIAL DEGREE $(TOT_DGRE)")
println("CUTOFF $(r_cut) Ang")
println("ENERGY WEIGHT: $(e_wght)")
println("FORCE WEIGHT: $(f_wght)")


data = read_extxyz(data_file)
data_shuffled = shuffle(data)
split_ind = floor(Int,length(data)*0.8)
train_data = data_shuffled[1:split_ind]
test = data_shuffled[split_ind:end]

# train_data = data[data_ids["train"].+1]
# test = data[data_ids["test"].+1] 

# save indices of test and train
train_idx = [] 
test_idx = []
for (idx, dp) in enumerate(data)
    if dp in test
        push!(test_idx,Int(idx-1))
    elseif dp in train_data
        push!(train_idx,Int(idx-1))
    end
end



# data_ids = load("../../0_splits/split$(split_no).jld2")
# # data = read_extxyz(data_file)

# train_data = data_ids["train"]
# test = data_ids["test"]

basis = ACE1x.ace_basis(elements = [:Cu, :H],
                order = length(TOT_DGRE),
                totaldegree = TOT_DGRE,
                rcut = r_cut)

println("Length of the basis: $(length(basis))")


weights = Dict(
    "default" => Dict("E" => e_wght, "F" => f_wght, "V" => 0.0 ))

train = [ACEpotentials.AtomsData(t; energy_key="energy", force_key="forces", virial_key="virial", weights=weights) for t in train_data]
A, Y, W = ACEfit.assemble(train, basis)

solver = ACEfit.BLR()
results = ACEfit.solve(solver, W .* A, W .* Y)
model = JuLIP.MLIPs.SumIP(JuLIP.MLIPs.combine(basis, results["C"]))

test_res = [ACEpotentials.AtomsData(t; weights=weights) for t in test]

@info("Test Error Tables")
@info("First Potential: ")
ACEpotentials.linear_errors(test_res, model);



mkpath("../models_out/$(ver)/")
save_potential("../models_out/$(ver)/$(cur_name).json", model)
JLD2.save("../models_out/$(ver)/$(cur_name).jld2", Dict("test"=>test_idx, "train"=>train_idx))
npzwrite("../models_out/$(ver)/$(cur_name).npz", Dict("test" => convert(Array{Int32,1}, test_idx), "train" => convert(Array{Int32,1}, train_idx)))
export2lammps("../models_out/$(ver)/$(cur_name).yace", model)