---
layout: default
title: SchNet
parent: MLIP calculators
nav_order: 5
---

# SchNet calculator

[SchNet](https://github.com/atomistic-machine-learning/schnetpack) is one of the most popular message-passing neural-network-based MLIP.

Below are the instructions on how to initialize the SchNet calculator, to run dynamics simulations within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) using [ASE interface](https://nqcd.github.io/NQCDynamics.jl/stable/NQCModels/ase/).

{: .warning }
The following instructions will include **Julia**-based code.

We start with importing NQCDynamics.jl packages and PyCall which allows importing Python-based packages.

```jl
using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
spk_utils = pyimport("schnetpack.utils")
spk_interfaces = pyimport("schnetpack.interfaces")
```

Now, we specify paths to the model and atoms objects. Then we read the ASE Atoms object and we convert it to NQCDynamics object.

```jl
pes_model_path = "path/to/schnet/model/best_model"
atoms_path = "path/to/atoms.xyz"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)
```

We then set up our SchNet calculator and create NQCModels [AdiabaticASEModel](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCModels/adiabaticmodels/#NQCModels.AdiabaticModels.AdiabaticASEModel) object that includes the model.

```jl
spk_model = spk_utils.load_model(pes_model_path;map_location="cpu")
model_args = spk_utils.read_from_json("$(pes_model_path)/args.json")
environment_provider = spk_utils.script_utils.settings.get_environment_provider(model_args,device="cpu")
calculator = spk_interfaces.SpkCalculator(spk_model, energy="energy", forces="forces", environment_provider=environment_provider)
ase_atoms.set_calculator(calculator)
pes_model = AdiabaticASEModel(ase_atoms)
```


Finally, we can use the model to e.g. initialize [Simulation](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCDynamics/nonadiabaticmoleculardynamics/#NQCDynamics.Simulation-Union%7BTuple%7BT%7D,%20Tuple%7BM%7D,%20Tuple%7BAtoms%7BT%7D,%20NQCModels.Model,%20M%7D%7D%20where%20%7BM,%20T%7D) object that is employed to run MD simulations.

```jl
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
```


# References

[K. Schütt and P.-J. Kindermans and F. Sauceda, E. Huziel and S. Chmiela and A. Tkatchenko and K.-R. Müller, SchNet: A continuous-filter convolutional neural network for modeling quantum interactions, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)

[K.T. Schütt, H.E. Sauceda, P.-J. Kindermans, A. Tkatchenko, K.-R. Müller, SchNet – A deep learning architecture for molecules and materials, J. Chem. Phys., 148, 241722, 2018](https://doi.org/10.1063/1.5019779)