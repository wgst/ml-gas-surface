using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
reann = pyimport("ase.calculators.reann")

"""
Function for creating ASE object from REANN model
"""
function reann_model_pes(model_path, cur_atoms, atomtype, period)
    calculator = reann.REANN(
        device="cpu", 
        atomtype=atomtype, 
        period=period, 
        nn=model_path)
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end


################################################
################### USE MODEL ##################
################################################
pes_model_path = "../../../models/reann/REANN_PES_DOUBLE.pt"
atoms_path = "../../../dbs/example_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)
atomtype = ["Cu","H"]
period = [1,1,1]

println("Load ML models and initialize the simulation...")
pes_model = reann_model_pes(pes_model_path, ase_atoms, atomtype, period)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
