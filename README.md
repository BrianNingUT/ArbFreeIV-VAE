# ArbFreeIV-VAE
Code repository for demos of the article 'Arbitrage-Free Implied Volatility Surface Generation with Variational Autoencoders'. A preprint of the article can be found at: https://arxiv.org/abs/2108.04941 .

This repository contains files used to generate some of the figures found in the paper, a short demo on how to fit the CTMC-SDE (CTMC) model, and how to fit the CTMC_VAE using some precomputed outputs. 

**IMPORTANT NOTES**
- Due to data constraints, only a single day's implied volatility surface data has been provided however the demo can be run on other data if provided. 
- For simplicity all code has been ported into Python, however this significantly increases the run time of some notebooks, in particular CTMC_Model_Fitting.ipynb. 

**Description of files**
Notebooks: Detailed descriptions are provide inside of each.
- CTMC_Model_Fitting.ipynb:            Fitting of the CTMC model on a single day's IV data
- CTMC_VAE_Fit.ipynb:                  Fitting of the CTMC-VAE model on precomputed CTMC model parameters.
- Pairwise_param_scatter.ipynb:        Scatter plots of generated parameters of the CTMC-VAE model (Figure 5 in article)
- Random_Surfaces.ipynb:               Several randomly generated surfaces for different currency pairs (Figure 7 in article)

Python files:
- ctmc.py:                             Functions pertaining to computation of the price and densities of the CTMC-SDE model.
- DensityEstimation.py:                Functions pertaining to computation of the spline implied density of the CTMC-SDE model.
- Fit_CTMC.py:                         Functions pertaining to fitting the CTMC-SDE model.
- VAE_fit.py:                          Functions pertaining to generation and fitting of the CTMC_VAE model.
- helpers.py:                          General helper functions

Data/Networks:
- all_cur_train_valid_days_new.pickle  Contains the selected training and testing days.
- ###_fitted_params.pickle 
