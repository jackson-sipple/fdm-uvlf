# FDM UVLF

Code for the paper "Fuzzy Dark Matter Constraints from the Hubble Frontier Fields."
This repo contains the UV luminosity function modeling pipeline: conditional luminosity
function models, MCMC fitting via emcee, and plotting utilities.

## Quick start

```bash
pip install -r requirements.txt
```

Then open `demo_handoff.ipynb` for a walkthrough of how to plug in your own halo mass
function and run the optimizer/MCMC pipeline.

## Repository layout

- `lf_model.py` — Model classes (LFModel, FiducialCLF, ReciprocalFuzzyCLF, etc.)
- `lf_optimizer.py` — Optimizer and MCMC runner (wraps emcee)
- `lf_plotter.py` — Plotting utilities for fits, corner plots, etc.
- `mass_function.py` — Halo mass function computation (uses CAMB)
- `utils.py` — Shared constants, interpolation helpers, cosmology
- `dust_correction.py` — Dust correction applied to observed magnitudes
- `P_k.npz` — Precomputed matter power spectrum from CAMB
- `mass_fns.npz` — Precomputed CDM halo mass functions at z=0..19
- `lf/by_name/` — Measurement data and best-fit params for each model variant
- `reference_figs/` — Reference output figures from the paper
- `newfdm6.1_jan6.ipynb`, `newfdm6_jan9.ipynb` — Original analysis notebooks (may
  not run end-to-end without additional data files; kept for reference)

## What's NOT in this repo (and how to regenerate)

| Artifact | How to regenerate |
|----------|-------------------|
| `mcmc.h5` chains | `lf_optimizer.run_mcmc([dir], ModelClass=..., n_steps=5000)` |
| Marsh HMF tables (`inv_log_spacing500_to_z20.npz`) | Loop `mass_function.MassFunction(m_FDM=..., is_marsh=True).dndM()` over a grid of masses/redshifts |
| Sharp-k HMF tables | Same approach with `window='sharp k'` |
| Jeffreys prior files (`jefferys_prior_*.npz`) | Only needed for Jeffreys-prior model variants |

## Known compatibility notes

- Requires `scipy < 1.14` because `scipy.misc.derivative` and `scipy.interpolate.interp2d`
  were removed in scipy 1.14.
- Originally developed with Python 3.7/3.9; tested to work with 3.9+.
