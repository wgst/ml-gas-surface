---
layout: default
title: ML interatomic potentials
nav_order: 3
---

# Training MLIPs
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

# Introduction
Traditionally, we utilize *ab-initio* methods, within density functional theory (DFT), to run molecular dynamics simulations. However, for many systems or problems these methods are too computationally expensive, e.g. due to the long length of the simulation or to the high number of trajectories. This is often the case within the gas-surface systems, for which many experimental results provide reaction probabilities. To model such probabilities, high number of trajectories is required. One of the most popular method that allows significantly faster force evaluations is machine learning (ML).

# Machine learning interatomic potentials
Employing machine learning interatomic potentials (MLIPs) in our dynamics can reduce the simulation times by several orders of magnitude. There are many MLIP methods, that mainly differ by descriptor type or architecture.

Some of the most popular MLIPs include:
* [SchNet](https://github.com/atomistic-machine-learning/schnetpack/tree/master)
* [PaiNN](https://github.com/atomistic-machine-learning/schnetpack/tree/masters)
* [linear atomic cluster expansion (ACE)](https://github.com/ACEsuit/ACEpotentials.jl)
* [MACE](https://github.com/ACEsuit/mace/tree/main/mace)
* [Recursively embedded atom neural network (REANN)](https://github.com/zhangylch/REANN/tree/main)
* [NequIP](https://github.com/mir-group/nequip/tree/main)
* and many, many more!

