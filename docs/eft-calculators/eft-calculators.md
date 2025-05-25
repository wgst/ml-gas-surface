---
layout: default
title: EFT calculators
nav_order: 14
has_children: true
permalink: /docs/eft-calculators
---

# Electronic friction calculators

To run molecular dynamics with electronic friction, we need a ground state potential as described in [MLIP-calculators](https://wgst.github.io/ml-gas-surface/mlip-calculators/mlip-calculators.html) page. However, to include nonadiabatic effects in our dynamics, we also need electronic friction tensor (EFT). Within [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) environment this can be done using EFT connector [FrictionProviders.jl](https://github.com/NQCD/FrictionProviders.jl). [FrictionProviders.jl](https://github.com/NQCD/FrictionProviders.jl) allows connecting EFT ML models or cube files to molecular dynamics code for orbital dependent friction (ODF) and local density friction approximation (LDFA). In this section, we will show examples of including different [FrictionProviders.jl](https://github.com/NQCD/FrictionProviders.jl) calculators into [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl) dynamical infrastructure.

The instructions on loading EFT models are shown here:
* [ODF](https://wgst.github.io/ml-gas-surface/eft-calculators/eft-odf.md)
* [LDFA](https://wgst.github.io/ml-gas-surface/eft-calculators/eft-ldfa.md)
