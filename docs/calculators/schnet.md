---
layout: default
title: SchNet
parent: Calculators
nav_order: 4
---



## SchNet

```jl
using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
spk_utils = pyimport("schnetpack.utils")
spk_interfaces = pyimport("schnetpack.interfaces")
```

Function for creating ASE object from SchNet model

```jl
function schnet_model_pes(model_path, cur_atoms)
    spk_model = spk_utils.load_model(model_path;map_location="cpu")
    model_args = spk_utils.read_from_json("$(model_path)/args.json")
    environment_provider = spk_utils.script_utils.settings.get_environment_provider(model_args,device="cpu")
    calculator = spk_interfaces.SpkCalculator(spk_model, energy="energy", forces="forces", environment_provider=environment_provider)
    cur_atoms.set_calculator(calculator)
    model = AdiabaticASEModel(cur_atoms)

    return model
end
```

Using the model within NQCDynamics.jl

```jl
pes_model_path = "../../../models/schnet/best_model"
atoms_path = "../../../dbs/example_structures/cu111_h7.0_full_925K.in"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)
```

Loading ML models and initializing the simulation

```jl
pes_model = schnet_model_pes(pes_model_path, ase_atoms)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
```

