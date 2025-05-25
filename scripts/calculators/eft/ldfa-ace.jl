using FrictionProviders
using PyCall: pyimport
using NQCBase: NQCBase
using Unitful: @u_str
using ACE1
using ASE
using JuLIP
using NQCModels

# ACE MODEL
io = pyimport("ase.io")


"""
Function for creating ACE model object
"""
function ace_model_density(model_path, atoms_ase)
    IP = ACE1.read_dict(load_dict(model_path)["IP"])
    JuLIP.set_calculator!(atoms_ase, IP)
    model = AdiabaticModels.JuLIPModel(atoms_ase)

    return model
end

# WORKS WITH PACKAGE VERSIONS:
# [e3f9bc04] ACE1 v0.12.0
# [3b96b61c] ACEpotentials v0.6.5
# [945c410c] JuLIP v0.14.6
# [3b96b61c] ACEpotentials v0.6.5
# [c814dc9f] NQCModels v0.8.20

model_p = "../../../models/h2cu/density-eft-ldfa/h2cu_ace.json"
atoms_p = "../../dynamics/state-to-state-scattering/initial_conditions/input_structures/cu111_h7.0_full_925K.in"
atoms_ase = io.read(atoms_p)
atoms, R, cell =  NQCBase.convert_from_ase_atoms(atoms_ase)
density_unit = u"Å^-3"
eft_ids = [length(atoms_ase)-1,length(atoms_ase)] # remember that Cu(211) in my db is 3x4 (other facets are 3x3), so 'atoms_ase' may not always have 56 atoms

atoms_ase.pop() # for density models we need a structure with a single H atom for model initialization - make sure 'atoms' still includes 2 atoms
atoms_ase_jl = ASE.ASEAtoms(atoms_ase)
atoms_julip = JuLIP.Atoms(atoms_ase_jl)

ace_model = ace_model_density(model_p, atoms_julip)
density_model = AceLDFA(ace_model; density_unit=density_unit)

model = LDFAFriction(density_model, atoms; friction_atoms=eft_ids)