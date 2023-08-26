---
layout: default
title: Training MLIPs
nav_order: 3
---


In order to run MD simulation, we have to connect our ML model with the MD infrastructure. This is done through the calculators and depending on a MLIP, there are different types of calculators. The most popular calculators are used within [ASE](https://wiki.fysik.dtu.dk/ase/). Many codes enable connecting the MLIP to LAMMPS. In our case, we use ASE calculators that we employ within [https://github.com/NQCD/NQCDynamics.jl](NQCDynamics.jl) to run the dynamics. Below, we will show examples of including SchNet and PaiNN calculators into [https://github.com/NQCD/NQCDynamics.jl](NQCDynamics.jl) dynamical infrastructure.

Instructions on how to run the MD simulations within the [https://github.com/NQCD/NQCDynamics.jl](NQCDynamics.jl) can be found here:
* [Classical molecular dynamics](https://nqcd.github.io/NQCDynamics.jl/dev/dynamicssimulations/dynamicsmethods/classical/)
* [Reactive scattering from a metal surface](https://nqcd.github.io/NQCDynamics.jl/dev/examples/reactive_scattering/)


Examples for MLIP calculators:
* SchNet
* PaiNN

