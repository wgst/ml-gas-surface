---
layout: default
title: Adaptive sampling
nav_order: 7
has_children: true
permalink: /docs/adaptive_sampling
has_toc: false
---

# Adaptive sampling
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}

# Introduction
The adaptive sampling procedure for gas-surface dynamics described below is based on [Stark et al.](https://arxiv.org/abs/2305.10873) approach. The scheme for the adaptive sampling procedure is shown below.
<img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/figures/adaptive_sampling_scheme.png?raw=true" width="400">

# Initial database
The entire process of building a database for ML-based interatomic potentials (MLIPs) starts with generating initial database. This is usually done by running several *ab initio* molecular dynamics (AIMD) trajectories with different settings/systems. For example, [Stark et al.](https://arxiv.org/abs/2305.10873) start with running AIMD trajectories in different temperatures and for 4 different Cu facets.

After the AIMD simulations are done, we can add every n-th structure of each trajectory in our initial database (where n is mainly dependent on simulation step size and atomic environment).

# Training ML-based interatomic potentials
The second step of adaptive sampling is MLIP training. We need at least 3 models for energy error evaluation. The models can differ e.g. by training on the same database, but different random training-validation-test sets.

{: .note }
Training of MLIP is different with every method. We list some of the most popular methods with link to the repositories in the [**ML interatomic potentials**](https://wgst.github.io/ml-gas-surface/mlips.html) page.

Training is usually not straightforward and should include proper optimization of hyperparameters (k-fold cross-validation).

# High-error search within molecular dynamics
The trained MLIPs can now be used to search for the high-error structures in our chemical space. This is done by running MD simulations using one of the models and evaluating energy at every or every n-th step with all of the trained models, to calculate error (standard deviation) of the predictions. High-error structures are then saved in a database.

{: .note }
We include a more detailed description of this step, together with an explanation of scripts in the [**High-error structure search**](https://wgst.github.io/ml-gas-surface/adaptive_sampling/high_error_structure_search.html) page.

# Structure selection (clustering)
The obtained high-error structures database is usually too sizeable for just including it in our main database in its entirety. One of the most popular approach to select the most diverse, and thus the most informative, data points is k-means clustering. Within this method, we can choose how many clusters of our data do we need and then we establish centers of these clusters to obtain our final filtered high-error structure database.

{: .note }
We include a more detailed description of this step, together with explanation of scritps in  the [**Structure selection**](https://wgst.github.io/ml-gas-surface/adaptive_sampling/clustering.html) page.


Following the structure selection, energy, forces, and any other important properties can be calculated using electronic structure codes (usually within DFT). Then, the structures can be added to our initial database and the entire process can be repeated until we are satisfied with the obtained models.