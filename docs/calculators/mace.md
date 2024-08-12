---
layout: default
title: MACE
parent: MLIP calculators
nav_order: 8
---

# MACE calculator

[MACE](https://github.com/ACEsuit/mace) is an equivariant message-passing neural-network-based MLIP.

Below are the instructions on how to initialize the MACE calculator, to run dynamics simulations within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) using [ASE interface](https://nqcd.github.io/NQCDynamics.jl/stable/NQCModels/ase/).

{: .warning }
The following instructions will include **Julia**-based code.

We start with importing NQCDynamics.jl packages and PyCall which allows importing Python-based packages.

```jl
using NQCDynamics
using PyCall
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
mace_calc = pyimport("mace.calculators")
```


Now, we specify the cutoff distance, paths to the model, and Atoms objects. Then we read the ASE atoms object and we convert it to NQCDynamics object.

```jl
pes_model_path = "path/to/ace/model/MACE_model_swa.model"
atoms_path = "path/to/atoms.xyz"
ase_atoms = io.read(atoms_path)
atoms, positions, cell = NQCDynamics.convert_from_ase_atoms(ase_atoms)
```


We then set up our MACE calculator and create NQCModels [AdiabaticASEModel](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCModels/adiabaticmodels/#NQCModels.AdiabaticModels.AdiabaticASEModel) object that includes the model.

```jl
calculator = mace_calc.MACECalculator(
    model_path=pes_model_path, 
    device="cpu", 
    default_dtype="float32") # device = "cpu" or "cuda"
ase_atoms.set_calculator(calculator)
pes_model = AdiabaticASEModel(ase_atoms)
```

Finally, we can use the model to e.g. initialize [Simulation](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCDynamics/nonadiabaticmoleculardynamics/#NQCDynamics.Simulation-Union%7BTuple%7BT%7D,%20Tuple%7BM%7D,%20Tuple%7BAtoms%7BT%7D,%20NQCModels.Model,%20M%7D%7D%20where%20%7BM,%20T%7D) object that is employed to run MD simulations.

```jl
sim = Simulation{Classical}(atoms, pes_model, cell=cell)
```


## References

[K. Schütt, O. Unke, M. Gastegger, Equivariant message passing for the prediction of tensorial properties and molecular spectra, PMLR 2021](https://proceedings.mlr.press/v139/schutt21a.html)
