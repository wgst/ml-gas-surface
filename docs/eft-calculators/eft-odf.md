---
layout: default
title: EFT-ODF
parent: EFT calculators
nav_order: 16
---

# EFT-ODF calculator

Orbital dependent friction (ODF) allows evaluating electronic friction tensors (EFTs) from electronic structure code, enabling evaluation of full EFTs. For more details, go to [NQCDynamics-ODF](https://nqcd.github.io/NQCDynamics.jl/stable/dynamicssimulations/dynamicsmethods/mdef/#Time-dependent-Perturbation-theory-(TDPT)).


Below are the instructions on how to initialize the [ACEds](https://github.com/ACEsuit/ACEds.jl) (currently: [ACEfriction](https://github.com/ACEsuit/ACEfriction.jl)) ODF calculator, to run molecular dynamics with electronic friction (MDEF) within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) using [FrictionProviders.jl](https://github.com/NQCD/FrictionProviders.jl).

{: .warning }
The following instructions will include **Julia**-based code.

We start with importing NQCDynamics.jl packages and PyCall which allows importing Python-based packages.

```jl
using NQCDynamics
using PyCall
using NQCModels
using Unitful
using UnitfulAtomic
using FrictionProviders
using NQCBase: NQCBase
using ACE
using ACEds: ac_matrixmodel
using ACEds.FrictionModels
using ACEds.FrictionModels: Gamma
using JuLIP
using JuLIP: set_positions!
using ASE
using JLD2
# Importing Python modules with PyCall
io = pyimport("ase.io")
```


Now, we specify the EFT units and indices, and paths to the model. We read the ASE atoms object and we convert it to [JuLIP](https://github.com/JuliaMolSim/JuLIP.jl) Atoms object.

```jl
eft_unit = u"ps^-1"
friction_ids = [length(ase_atoms)-1,length(ase_atoms)]
eft_model_path = "path/to/aceds/model/eft_ac.model"
atoms_path = "path/to/atoms.xyz"
ase_atoms = io.read(atoms_path)
ase_jl = ASE.ASEAtoms(ase_atoms)
julip_atoms = JuLIP.Atoms(ase_jl)
```

Next, we load the [ACEds](https://github.com/ACEsuit/ACEds.jl) model and we create ODFriction object with [FrictionProviders.jl](https://github.com/NQCD/FrictionProviders.jl).

```jl
eft_model_aceds = read_dict(load_dict(eft_model_path))
aceds_model = ACEdsODF(eft_model_aceds, Gamma, julip_atoms; friction_unit=eft_unit)     
odf_model = ODFriction(aceds_model; friction_atoms=friction_ids)
```

Together with PES model, the above ODF object (ODFriction) can be then used for MDEF simulations (as documented in [NQCDynamics-MDEF](https://nqcd.github.io/NQCDynamics.jl/stable/dynamicssimulations/dynamicsmethods/mdef/)).

## References

[J. Gardner, O. A. Douglas-Gallardo, W. G. Stark, J. Westermayr, S. M. Janke, S. Habershon, R. J. Maurer, NQCDynamics.jl: A Julia package for nonadiabatic quantum classical molecular dynamics in the condensed phase, J. Chem. Phys. 156, 174801 (2022)](https://doi.org/10.1063/5.0089436)
