---
layout: default
title: Calculators
nav_order: 2
---


In order to run MD simulation, we have to connect our ML model with the MD infrastructure. This is done through the calculators and depending on a MLIP, there are different types of calculators. The most popular calculators are used within [ASE](https://wiki.fysik.dtu.dk/ase/). Many codes enable connecting the MLIP to LAMMPS. In our case, we use ASE calculators that we employ within [https://github.com/NQCD/NQCDynamics.jl](NQCDynamics.jl) to run the dynamics. Below, we will show examples of including SchNet and PaiNN calculators into [https://github.com/NQCD/NQCDynamics.jl](NQCDynamics.jl) dynamical infrastructure.

Instructions on how to run the MD simulations within the [https://github.com/NQCD/NQCDynamics.jl](NQCDynamics.jl) can be found here:
* [Classical molecular dynamics](https://nqcd.github.io/NQCDynamics.jl/dev/dynamicssimulations/dynamicsmethods/classical/)
* [Reactive scattering from a metal surface](https://nqcd.github.io/NQCDynamics.jl/dev/examples/reactive_scattering/)

## SchNet

```jl
using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
spk_utils = pyimport("schnetpack.utils")
spk_interfaces = pyimport("schnetpack.interfaces")

"""
Function for creating ASE object from SchNet model
"""
function schnet_model_pes(model_path, cur_atoms)
    spk_model = spk_utils.load_model(model_path;map_location="cpu")
    model_args = spk_utils.read_from_json("$(model_path)/args.json")
    environment_provider = spk_utils.script_utils.settings.get_environment_provider(model_args,device="cpu")
    calculator = spk_interfaces.SpkCalculator(spk_model, energy="energy", forces="forces", environment_provider=environment_provider)
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end


################################################
################### USE MODEL ##################
################################################
pes_model_path = "../../../models/schnet/best_model"
atoms_path = "../../../dbs/example_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)

println("Load ML models and initialize the simulation...")
pes_model = schnet_model_pes(pes_model_path, ase_atoms)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
```








## PaiNN

```jl
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
```