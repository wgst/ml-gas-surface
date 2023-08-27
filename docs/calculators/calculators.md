---
layout: default
title: MLIP calculators
nav_order: 4
has_children: true
permalink: /docs/calculators
---


In order to run MD simulation, we have to connect our ML model with the MD infrastructure. This is done through the calculators and depending on a MLIP, there are different types of calculators. The most popular calculators are used within [ASE](https://wiki.fysik.dtu.dk/ase/). Many codes also enable connecting the MLIPs to LAMMPS. In our case, we use ASE calculators that we employ within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl), dynamics code. In this section, we will show examples of including different calculators into [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) dynamical infrastructure.

Instructions on how to run the MD simulations within the [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) can be found here:
* [Classical molecular dynamics](https://nqcd.github.io/NQCDynamics.jl/dev/dynamicssimulations/dynamicsmethods/classical/)
* [Reactive scattering from a metal surface](https://nqcd.github.io/NQCDynamics.jl/dev/examples/reactive_scattering/)
