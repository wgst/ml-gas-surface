<p align="center">
  <img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/figures/ml_to_gas_surf.png?raw=true" width="300">
</p>


# ML for gas-surface dynamics

Instructions and scripts for growing machine learning interatomic potentials (MLIPs) databases through adaptive sampling for gas-surface dynamics.
* **High-error structure search** using [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl).
* **Structure selection (clustering)** for database reduction.

## [**>>> Instructions are available here <<<**](https://wgst.github.io/ml-gas-surface/)

## References
**If you found the scripts and/or tutorial helpful, please cite the following references:**

W. G. Stark, C. L. Box, M. Sachs, N. Hertl, R. J. Maurer, Nonadiabatic reactive scattering of hydrogen on different surface facets of copper, arXiv:2505.18147 [physics.chem-ph], (2025) [[arXiv]](http://arxiv.org/abs/2505.18147)

```text
@misc{stark_nonadiabatic_2025,
	title = {Nonadiabatic reactive scattering of hydrogen on different surface facets of copper},
	doi = {10.48550/arXiv.2505.18147},
	publisher = {arXiv},
	author = {Stark, Wojciech G. and Box, Connor L. and Sachs, Matthias and Hertl, Nils and Maurer, Reinhard J.},
	year = {2025},
	note = {arXiv:2505.18147 [physics.chem-ph]},
	url = {http://arxiv.org/abs/2505.18147},
}
```

W. G. Stark, J. Westermayr, O. A. Douglas-Gallardo, J. Gardner, S. Habershon, R. J. Maurer, Machine learning interatomic potentials for reactive hydrogen dynamics at metal surfaces based on iterative refinement of reaction probabilities, J. Phys. Chem. C, 127, 50, 24168–24182, (2023) [[arXiv]](https://arxiv.org/abs/2305.10873) [[journal]](https://pubs.acs.org/doi/10.1021/acs.jpcc.3c06648)

```text
@article{stark_machine_2023,
	title = {Machine learning interatomic potentials for reactive hydrogen dynamics at metal surfaces based on iterative refinement of reaction probabilities},
	author = {Stark, Wojciech G. and Westermayr, Julia and Douglas-Gallardo, Oscar A. and Gardner, James and Habershon, Scott and Maurer, Reinhard J.},
	volume = {127},
	number = {50},
	pages = {24168–24182},
	year = {2023},
	publisher = {J. Phys. Chem. C},
	doi = {10.1021/acs.jpcc.3c06648}, 
	url = {https://doi.org/10.1021/acs.jpcc.3c06648}
}
```

J. Gardner, O. A. Douglas-Gallardo, W. G. Stark, J. Westermayr, S. M. Janke, S. Habershon, R. J. Maurer, NQCDynamics.jl: A Julia package for nonadiabatic quantum classical molecular dynamics in the condensed phase, J. Chem. Phys. 156, 174801 (2022) [[arXiv]](https://arxiv.org/abs/2202.12925) [[journal]](https://doi.org/10.1063/5.0089436)

```text
@article{gardner_nqcdynamicsjl_2022,
	title = {{NQCDynamics}.jl: {A} {Julia} package for nonadiabatic quantum classical molecular dynamics in the condensed phase},
	author = {Gardner, James and Douglas-Gallardo, Oscar A. and Stark, Wojciech G. and Westermayr, Julia and Janke, Svenja M. and Habershon, Scott and Maurer, Reinhard J.},
	journal = {J. Chem. Phys.},
	volume = {156},
	number = {17},
	pages = {174801},
	year = {2022},
	issn = {0021-9606, 1089-7690},
	doi = {10.1063/5.0089436},
	url = {https://aip.scitation.org/doi/10.1063/5.0089436}
}
```
