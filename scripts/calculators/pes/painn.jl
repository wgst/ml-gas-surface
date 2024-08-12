using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
spk_ase_interface = pyimport("schnetpack.interfaces.ase_interface")
spk_transform = pyimport("schnetpack.transform")

"""
Function for creating ASE object from PaiNN model (SchNetPack 2.0)
"""
function painn_model_pes(model_path, cur_atoms, cutoff)
    calculator = spk_ase_interface.SpkCalculator(
        model_file=model_path,
        stress_key="stress",
        neighbor_list=spk_transform.ASENeighborList(cutoff=cutoff),
        energy_unit="eV",
        forces_units="eV/Angstrom",
        stress_units="eV/Ang/Ang/Ang"
    )
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end


################################################
################### USE MODEL ##################
################################################
pes_model_path = "../../../models/painn/best_inference_model"
atoms_path = "../../../dbs/example_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)
cutoff = 4.0  # Angstrom (units used in model)

println("Load ML models and initialize the simulation...")
pes_model = painn_model_pes(pes_model_path, ase_atoms, cutoff)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
