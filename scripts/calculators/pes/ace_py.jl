using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
pyjulip = pyimport("pyjulip")

"""
Function for creating ASE object from ACE model
"""
function ace_model_pes(model_path, cur_atoms) # cur_atoms has to be JuLIP atoms object
    calculator = pyjulip.ACE1(model_path)
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end


################################################
################### USE MODEL ##################
################################################
pes_model_path = "../../../models/ace/h2cu_ace.json"
atoms_path = "../../../dbs/example_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)

println("Load ML models and initialize the simulation...")
pes_model = ace_model_pes(pes_model_path, ase_atoms)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
