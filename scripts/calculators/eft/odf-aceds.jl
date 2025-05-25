using NQCDynamics
using PyCall
using NQCModels
using Unitful
using UnitfulAtomic
using FrictionProviders
using NQCBase: NQCBase
using ACE
using ACEds: ac_matrixmodel
using ACEds.FrictionModels
using ACEds.FrictionModels: Gamma
using JuLIP
using JuLIP: set_positions!
using ASE
using JLD2
# Importing Python modules with PyCall
io = pyimport("ase.io")


##### EFT MODEL #####
eft_model_path = "../../../models/h2cu/eft-odf/eft_ac.model"
eft_unit = u"ps^-1"

atoms_path = "../../dynamics/state-to-state-scattering/initial_conditions/input_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
ase_jl = ASE.ASEAtoms(ase_atoms)
julip_atoms = JuLIP.Atoms(ase_jl)
friction_ids = [length(ase_atoms)-1,length(ase_atoms)]

# LOAD EFT MODEL    
println("Load ML models...")
eft_model_aceds = read_dict(load_dict(eft_model_path))
aceds_model = ACEdsODF(eft_model_aceds, Gamma, julip_atoms; friction_unit=eft_unit)     
odf_model = ODFriction(aceds_model; friction_atoms=friction_ids)