using NQCDynamics
using PyCall
using NQCModels
using ASE
using JuLIP
using ACE1

# Importing Python modules with PyCall
io = pyimport("ase.io")
pyjulip = pyimport("pyjulip")

"""
Function for creating ASE object from ACE model
"""
function ace_model_pes(model_path, cur_atoms) # cur_atoms has to be JuLIP atoms object
    IP = ACE1.read_dict(load_dict(model_path)["IP"])
    JuLIP.set_calculator!(cur_atoms, IP)
    model = AdiabaticModels.JuLIPModel(cur_atoms)

    return model
end


################################################
################### USE MODEL ##################
################################################
pes_model_path = "../../../models/ace/h2cu_ace.json"
atoms_path = "../../../dbs/example_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
ase_jl = ASE.ASEAtoms(ase_atoms)
atoms_julip = JuLIP.Atoms(ase_jl)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)

println("Load ML models and initialize the simulation...")
pes_model = ace_model_pes(pes_model_path, atoms_julip)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
