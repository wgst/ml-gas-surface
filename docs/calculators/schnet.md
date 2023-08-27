---
layout: default
title: SchNet
parent: MLIP calculators
nav_order: 5
---

[https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html](SchNet) is one of the most popular message-passing neural-network-based MLIPs.

Below are the instructions on how to initialize the SchNet calculator, to run dynamics simulations within [https://github.com/NQCD/NQCDynamics.jl](NQCDynamics.jl).


We start with importing NQCDynamics.jl packages and PyCall that allows importing python-based packages.

```jl
using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
spk_utils = pyimport("schnetpack.utils")
spk_interfaces = pyimport("schnetpack.interfaces")
```

We define a function for creating NQCModels object that includes SchNet model, using two variables: path to our SchNet model and ASE-based atoms object.

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

Now, we specify paths to the model and atoms objects. Then we read the ASE atoms object and we convert it to NQCDynamics object.

```jl
pes_model_path = "path/to/schnet/model/best_model"
atoms_path = "path/to/atoms.xyz"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)
```

Finally, we initialize the model using previously defined 'schnet_model_pes' function and we initialize Simulation object that can be used e.g. to run dynamics simulations.

```jl
pes_model = schnet_model_pes(pes_model_path, ase_atoms)
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
```


## References

[https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html](K. Schütt and P.-J. Kindermans and F. Sauceda, E. Huziel and S. Chmiela and A. Tkatchenko and K.-R. Müller, SchNet: A continuous-filter convolutional neural network for modeling quantum interactions, NeurIPS 2017)

[https://doi.org/10.1063/1.5019779](K.T. Schütt, H.E. Sauceda, P.-J. Kindermans, A. Tkatchenko, K.-R. Müller, SchNet – A deep learning architecture for molecules and materials, J. Chem. Phys., 148, 241722, 2018)