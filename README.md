# GP alignment of multi-slice VLBI light curves

Fits a shared latent curve `f(t)` across multiple time series, estimating per-timeseries delays, amplitude scales and offsets.

## Model

Each timeseries (flux at offset $r_i$) is modelled as:

$$\log flux_i(t) = a_i * f(t - \tau_i) + b_i + noise$$

- $f(t)$ is a Gaussian process (Matern-5/2 kernel, discretised on a grid)
- $\tau_i$ is the time delay (how late the signal arrives at radius $r_i$)
- $a_i$ scales the amplitude, $b_i$ shifts the baseline
- First timeseries is not shifted/scaled: $\tau_1 = 0$, $a_1 = 1$

Fitting is MAP estimation via L-BFGS with automatic differentiation (ForwardDiff).
Uncertainties on delays are from a Laplace approximation (Hessian at the MAP).

## Usage

- Install [Julia](https://julialang.org/downloads/) version 1.10
- Install all required packages:
```julia
using Pkg; Pkg.instantiate()
```
- Run code from Julia: `include("gp_alignment_free_delays.jl")`

## Input

`data.csv` with columns:
- `year` -- observation epoch (decimal year)
- `r_mas` -- radial offset from core (milliarcseconds)
- `flux` -- flux density (Jy/beam)
- `err` -- measurement uncertainty on flux

## Output

- `gp_alignment_summary.png` -- 5-panel summary figure
- `results.csv` -- fitted parameters per slice (tau, tau_std, a, b)
