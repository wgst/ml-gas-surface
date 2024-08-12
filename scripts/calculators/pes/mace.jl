using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
mace_calc = pyimport("mace.calculators")

"""
Function for creating ASE object from MACE model
"""
function mace_model_pes(model_path, cur_atoms)
    calculator = mace_calc.MACECalculator(
        model_path=model_path, 
        device="cpu", 
        default_dtype="float32") # device = "cpu" or "cuda"
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end


################################################
################### USE MODEL ##################
################################################
pes_model_path = "../../../models/mace/MACE_model_swa.model"
atoms_path = "../../../dbs/example_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)

println("Load ML models and initialize the simulation...")
pes_model = mace_model_pes(pes_model_path, ase_atoms)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
