---
layout: default
title: EFT-LDFA
parent: EFT calculators
nav_order: 15
---

# EFT-LDFA calculator

Local density friction approximation (LDFA) allows evaluation of electronic friction tensors (EFTs) by utilizing surface density at the position of adsorbate atom. For details, please go to: [NQCDynamics-LDFA](https://nqcd.github.io/NQCDynamics.jl/stable/dynamicssimulations/dynamicsmethods/mdef/#Local-density-friction-approximation-(LDFA)).

To obtain EFT with LDFA, we need to evaluate surface electronic density. This can be done using many methods. Here, we will focus on [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl) models.

Below are the instructions on how to initialize the ACE-LDFA calculator, to run molecular dynamics with electronic friction (MDEF) within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) using [FrictionProviders.jl](https://github.com/NQCD/FrictionProviders.jl).

{: .warning }
The following instructions will include **Julia**-based code.

We start with importing [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) packages and PyCall which allows importing Python-based packages.

```jl
using FrictionProviders
using PyCall: pyimport
using NQCBase: NQCBase
using Unitful: @u_str
using ACE1
using ASE
using JuLIP
using NQCModels

# Importing Python modules with PyCall
io = pyimport("ase.io")
```


Now, we specify the density units, EFT indices, and paths to the model. We read the ASE atoms object and we convert it to [JuLIP](https://github.com/JuliaMolSim/JuLIP.jl) Atoms object.

```jl
density_unit = u"Å^-3"
eft_ids = [length(atoms_ase)-1,length(atoms_ase)] # remember that Cu(211) in my db is 3x4 (other facets are 3x3), so 'atoms_ase' may not always have 56 atoms

model_p = "path/to/ace/model/h2cu_ace.json"
atoms_p = "path/to/atoms.xyz"
atoms_ase = io.read(atoms_p)
atoms, R, cell =  NQCBase.convert_from_ase_atoms(atoms_ase)
atoms_ase.pop() # for density models we need a structure with a single H atom for model initialization - make sure 'atoms' still includes 2 atoms
atoms_ase_jl = ASE.ASEAtoms(atoms_ase) # convert to ASE.jl object
atoms_julip = JuLIP.Atoms(atoms_ase_jl) # convert to julip object
```


We then set up our [JuLIP](https://github.com/JuliaMolSim/JuLIP.jl)-ACE calculator within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) and we create AceLDFA and LDFAFriction objects with [FrictionProviders.jl](https://github.com/NQCD/FrictionProviders.jl).

```jl
IP = ACE1.read_dict(load_dict(model_p)["IP"])
JuLIP.set_calculator!(atoms_julip, IP)
ace_model = AdiabaticModels.JuLIPModel(atoms_julip)

density_model = AceLDFA(ace_model; density_unit=density_unit)

model = LDFAFriction(density_model, atoms; friction_atoms=eft_ids)
```

Together with PES model, the above LDFA object (LDFAFriction) can be then used for MDEF simulations (as documented in [NQCDynamics-MDEF](https://nqcd.github.io/NQCDynamics.jl/stable/dynamicssimulations/dynamicsmethods/mdef/)).


## References

[J. Gardner, O. A. Douglas-Gallardo, W. G. Stark, J. Westermayr, S. M. Janke, S. Habershon, R. J. Maurer, NQCDynamics.jl: A Julia package for nonadiabatic quantum classical molecular dynamics in the condensed phase, J. Chem. Phys. 156, 174801 (2022)](https://doi.org/10.1063/5.0089436)
