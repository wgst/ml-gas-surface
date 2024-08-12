---
layout: default
title: ACE
parent: MLIP calculators
nav_order: 9
---

# ACE calculator

[ACEpotentials](https://github.com/ACEsuit/ACEpotentials.jl) is an atomic cluster expansion (ACE)-based linear MLIP.

Below are the instructions on how to initialize the ACEpotentials calculator, to run dynamics simulations within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) using [ASE interface](https://nqcd.github.io/NQCDynamics.jl/stable/NQCModels/ase/).

{: .warning }
The following instructions will include **Julia**-based code.

We start with importing NQCDynamics.jl packages and PyCall which allows importing Python-based packages.

```jl
using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
pyjulip = pyimport("pyjulip")
```


Now, we specify the cutoff distance, paths to the model, and Atoms objects. Then we read the ASE atoms object and we convert it to NQCDynamics object.

```jl
pes_model_path = "path/to/ace/model/h2cu_ace.json"
atoms_path = "path/to/atoms.xyz"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)
```


We then set up our ACEpotentials calculator and create NQCModels [AdiabaticASEModel](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCModels/adiabaticmodels/#NQCModels.AdiabaticModels.AdiabaticASEModel) object that includes the model.

```jl
calculator = pyjulip.ACE1(pes_model_path)
ase_atoms.set_calculator(calculator)
pes_model = AdiabaticASEModel(ase_atoms)
```

The model can be loaded also directly within julia:

```jl
using ASE
using JuLIP
using ACE1

IP = ACE1.read_dict(load_dict(model_path)["IP"])
JuLIP.set_calculator!(ase_atoms, IP)
pes_model = AdiabaticModels.JuLIPModel(ase_atoms)
```

Finally, we can use the model to e.g. initialize [Simulation](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCDynamics/nonadiabaticmoleculardynamics/#NQCDynamics.Simulation-Union%7BTuple%7BT%7D,%20Tuple%7BM%7D,%20Tuple%7BAtoms%7BT%7D,%20NQCModels.Model,%20M%7D%7D%20where%20%7BM,%20T%7D) object that is employed to run MD simulations.

```jl
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
```


## References

[R. Drautz, Atomic cluster expansion for accurate and transferable interatomic potentials, Phys. Rev. B Condens. Matter. 99, 014104, 2019](https://doi.org/10.1103/PhysRevB.99.014104)
[G. Dusson, M. Bachmayr, G. Csanyi, S. Etter, C. van der Oord, and C. Ortner, ACEpotentials.jl: A Julia implementation of the atomic cluster expansion, J. Chem. Phys. 159, 164101, 2023](https://doi.org/10.1063/5.0158783)