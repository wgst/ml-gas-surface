---
layout: default
title: "Nonadiabatic dynamics"
nav_order: 17
has_children: true
permalink: /docs/nonadiabatic-dynamics
---

# Nonadiabatic dynamics: molecular dynamics with electronic friction

Continuum of states of metal surfaces may lead to significant nonadiabatic effects, which are not accounted for in classical MD simulations. One method to include such methods, while employing Born-Oppenheimer, ground state PES is molecular dynamics with electronic friction (MDEF). In MDEF, frictional and random forces are included within the equation of motion, to account for nonadiabatic effects. 

To run MDEF simulation, we first load PES model. Here, we load a [MACE](https://github.com/ACEsuit/mace) model, just as in [MLIP-calculators](https://wgst.github.io/ml-gas-surface/mlip-calculators/mlip-calculators.html).

```jl
pes_model = mace_model(ml_model_path, ase_atoms)
calculator_pes = mace_calc.MACECalculator(model_path=ml_model_path, device="cpu", default_dtype="float32")
```

Then, ODF model can be loaded. Here, we load an LDFA model, as described in [EFT-LDFA](https://wgst.github.io/ml-gas-surface/eft-calculators/eft-ldfa.md).

```jl
friction_ids = [length(ase_atoms)-1,length(ase_atoms)]
eft_model_ml = ace_model(eft_model_path, atoms_julip)
density_model = AceLDFA(eft_model_ml; density_unit=u"Å^-3")
eft_model = LDFAFriction(density_model, atoms; friction_atoms=friction_ids)
```

Finally, we combine the MLIP, and EFT models using [NQCModels.FrictionModels.CompositeFrictionModel](https://nqcd.github.io/NQCDynamics.jl/stable/api/NQCModels/frictionmodels/#NQCModels.FrictionModels.CompositeFrictionModel).

```jl
model = CompositeFrictionModel(pes_model,eft_model)
sim = Simulation{MDEF}(atoms, model, cell=cell, temperature=parse(Float64,temp_surf)*u"K")
```

To run the MDEF simulation we can then execute the following line, using the constructed simulation (sim) object, using initial conditions included in *distribution* variable.
```jl
ensemble = Ensembles.run_dynamics(sim, (0.0, 3000*u"fs"), distribution; dt=0.1*u"fs",
                        ensemble_algorithm=EnsembleDistributed(), saveat=(0.0:austrip(0.1*u"fs"):austrip(3000*u"fs")))
```

More instructions on how to run the MDEF simulations and how to use the simulation output within the [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) can be found [here](https://nqcd.github.io/NQCDynamics.jl/stable/dynamicssimulations/dynamicsmethods/mdef/)).