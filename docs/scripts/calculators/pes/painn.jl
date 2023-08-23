using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
spk_interfaces = pyimport("schnetpack.interfaces")
spk_transform = pyimport("schnetpack.transform")
torch = pyimport("torch")

"""
Function for creating ASE object from SchNet model
"""
function painn_model_pes(model_path, cur_atoms, cutoff)
    best_model = torch.load(model_path,map_location=torch.device("cpu") ).to("cpu")
    converter = spk_interfaces.AtomsConverter(neighbor_list=spk_transform.ASENeighborList(cutoff=cutoff), dtype=torch.float32)
    calculator = spk_interfaces.SpkCalculator(model=best_model, converter=converter, energy_units="eV", forces_units="eV/Angstrom")
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
