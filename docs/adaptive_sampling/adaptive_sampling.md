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
The adaptive sampling procedure for gas-surface dynamics is described by [Stark et al.](https://arxiv.org/abs/2305.10873). We will briefly describe the process here, starting with the scheme shown below.
<img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/figures/adaptive_sampling_scheme.png?raw=true" width="500">


## Initial database

## Training MLIPs
In order to go to the next step of adaptive sampling, we need to train at least 3 MLIPs. Usually, this can be done by just training on different random training-validation-test splits.
